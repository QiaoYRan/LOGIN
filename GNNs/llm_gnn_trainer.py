import torch
from time import time
import numpy as np


from core.GNNs.gnn_utils import EarlyStopping, pick_nodes_random
from core.data_utils.load import load_data, load_gpt_preds
from core.utils import time_logger

LOG_FREQ = 10


class LLMGNNTrainer():

    def __init__(self, cfg, feature_type):
        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers
        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = feature_type
        self.epochs = cfg.gnn.train.epochs

        self.pl_rate = cfg.gnn.train.pl_rate
        self.pl_loss_weight = cfg.gnn.train.pl_alpha_for_loss
        
        # ! Load data
        data, num_classes = load_data(
            self.dataset_name, use_dgl=False, use_text=False, seed=self.seed)

        self.num_nodes = data.y.shape[0]
        self.num_classes = num_classes
        data.y = data.y.squeeze()
        # ! pick nodes for getting pseudo labels
        self.pl_mask, self.gold_mask = pick_nodes_random(data.train_mask, self.pl_rate)
        self.pseudo_labels = torch.zeros_like(data.y) # TO BE UPDATED

        # ! Init gnn feature
        topk = 3 if self.dataset_name == 'pubmed' else 5
        if self.feature_type == 'emb': # stored in folder embedding
            print("Loading features from folder embedding...")
            features = torch.load(f"embedding/{self.dataset_name}.pt").x# TODO
        elif self.feature_type == 'ogb':
            print("Loading OGB features...")
            features = data.x
        elif self.feature_type == 'TA':
            print("Loading pretrained LM features (title and abstract) ...")
            LM_emb_path = f"prt_lm/{self.dataset_name}/{self.lm_model_name}-seed{self.seed}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, 768)))
            ).to(torch.float32)
        elif self.feature_type == 'E':
            print("Loading pretrained LM features (explanations) ...")
            LM_emb_path = f"prt_lm/{self.dataset_name}2/{self.lm_model_name}-seed{self.seed}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, 768)))
            ).to(torch.float32)
        elif self.feature_type == 'P':
            print("Loading top-k prediction features ...")
            features = load_gpt_preds(self.dataset_name, topk)
        else:
            print(
                f'Feature type {self.feature_type} not supported. Loading OGB features...')
            self.feature_type = 'ogb'
            features = data.x

        self.features = features.to(self.device)
        self.data = data.to(self.device)

        # ! Trainer init
        use_pred = self.feature_type == 'P'

        if self.gnn_model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif self.gnn_model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        elif self.gnn_model_name == "MLP":
            from core.GNNs.MLP.model import MLP as GNN
        else:
            print(f"Model {self.gnn_model_name} is not supported! Loading MLP ...")
            from core.GNNs.MLP.model import MLP as GNN

        self.model = GNN(in_channels=self.hidden_dim*topk if use_pred else self.features.shape[1],
                         hidden_channels=self.hidden_dim,
                         out_channels=self.num_classes,
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         use_pred=use_pred).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of parameters: {trainable_params}")
        self.ckpt = f"output/{self.dataset_name}/{self.gnn_model_name}_dropout_{self.dropout}.pt"
        self.stopper = EarlyStopping(
            patience=cfg.gnn.train.early_stop, path=self.ckpt) if cfg.gnn.train.early_stop > 0 else None
        self.loss_func = torch.nn.CrossEntropyLoss()

        from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

    def _forward(self, x, edge_index):
        logits = self.model(x, edge_index)  # small-graph
        return logits

    def _pretrain(self):
        # ! Shared
        self.model.train()
        self.optimizer.zero_grad()
        # ! Specific
        logits = self._forward(self.features, self.data.edge_index)
        loss = self.loss_func(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        train_acc = self.evaluator(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_acc, logits[self.data.train_mask]

    def _retrain_with_pl(self):
        # ! Shared
        self.model.train()
        self.optimizer.zero_grad()
        # ! Specific
        logits = self._forward(self.features, self.data.edge_index)
        pseudo_labels = self.pseudo_labels.to(self.device)
        # gt_labels_for_pl_nodes = self.data.y[self.pl_mask]
        # pseudo_labels[pseudo_labels == -1] = gt_labels_for_pl_nodes[pseudo_labels == -1]
        pl_loss = self.loss_func(
            logits[self.pl_mask], pseudo_labels[self.pl_mask]).to(self.device)
        gold_loss = self.loss_func(
            logits[self.gold_mask], self.data.y[self.gold_mask]).to(self.device)
        loss = pl_loss * self.pl_loss_weight + gold_loss * (1 - self.pl_loss_weight)
        
        train_acc = self.evaluator(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss.item(), train_acc, logits[self.data.train_mask]
    
    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self._forward(self.features, self.data.edge_index)
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, logits

    def update_pseudo_labels_and_features(self, pseudo_labels, emb): # pseudo_labels: list
        pl_nodes_list = torch.nonzero(self.pl_mask).squeeze().tolist()  
        self.pseudo_labels = torch.zeros_like(self.data.y) 
        print(len(pl_nodes_list))
        print(len(pseudo_labels))
        indices_to_remove = []
        for i, label in enumerate(pseudo_labels):
            if label != -1:
                self.pseudo_labels[pl_nodes_list[i]] = label
            else:
                self.pl_mask[pl_nodes_list[i]] = False
                self.gold_mask[pl_nodes_list[i]] = True
                indices_to_remove.append(i)
        for index in reversed(indices_to_remove):
            del pl_nodes_list[index]
            emb = torch.cat((emb[:index, :], emb[index + 1:, :]), dim=0)
        emb = emb.to(self.device)
        print('emb_shape:', emb.shape)
        print('orig_features_shape:', self.features.shape)
        print('pl_nodes_list_len:', len(pl_nodes_list))
        mapping = torch.nn.Linear(emb.shape[1], self.features.shape[1]).to(self.device)
        print(emb.device)
        print(self.features.device)
        mapped_emb = mapping(emb)
        self.features[pl_nodes_list] = mapped_emb
            
    def update_pseudo_labels_and_features_new(self, pl_mask, pseudo_labels, emb): # pseudo_labels: list
            pl_nodes_list = torch.nonzero(pl_mask).squeeze().tolist()  
            self.pseudo_labels = torch.zeros_like(self.data.y) 
            self.pl_mask = pl_mask
            print(len(pl_nodes_list))
            print(len(pseudo_labels))
            indices_to_remove = []
            for i, label in enumerate(pseudo_labels):
                if label != -1:
                    self.pseudo_labels[pl_nodes_list[i]] = label
                else:
                    pl_mask[pl_nodes_list[i]] = False
                    self.gold_mask[pl_nodes_list[i]] = True
                    indices_to_remove.append(i)
            for index in reversed(indices_to_remove):
                del pl_nodes_list[index]
                emb = torch.cat((emb[:index, :], emb[index + 1:, :]), dim=0)
            emb = emb.to(self.device)
            print('emb_shape:', emb.shape)
            print('orig_features_shape:', self.features.shape)
            print('pl_nodes_list_len:', len(pl_nodes_list))
            mapping = torch.nn.Linear(emb.shape[1], self.features.shape[1]).to(self.device)
            print(emb.device)
            print(self.features.device)
            mapped_emb = mapping(emb)
            self.features[pl_nodes_list] = mapped_emb

    @time_logger
    def train(self, prt_sign=True):
        # ! Training
        pseudo_labels = self.pseudo_labels
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            if prt_sign == True:
                loss, train_acc, logits = self._pretrain()
                val_acc, test_acc, _ = self._evaluate()
            else:
                loss, train_acc, logits = self._retrain_with_pl()
                val_acc, test_acc, _ = self._evaluate()
        # TODO: maybe two seperated early stoppers needed
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            if epoch % LOG_FREQ == 0:
                print(
                    f'Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {loss:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}, ES: {es_str}')

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))

        return self.model

    @ torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, test_acc, logits = self._evaluate()
        print(val_acc, test_acc)
        print(
            f'[{self.gnn_model_name} + {self.feature_type}] ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        return logits, res
