import torch
from time import time
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

from core.GNNs.gnn_utils import EarlyStopping, pick_nodes_random, data_normalization
from core.data_utils.load import load_data, load_gpt_preds
from core.utils import time_logger

LOG_FREQ = 10


class LCGNNTrainer():

    def __init__(self, cfg, feature_type, data, num_classes):
        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.model.name
        self.llm_model_name = cfg.llm.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers
        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = feature_type
        self.epochs = cfg.gnn.train.epochs

        self.pl_rate = cfg.gnn.train.pl_rate
        self.pl_loss_weight = cfg.gnn.train.pl_alpha_for_loss
        self.remain_ratio = cfg.gnn.train.remain_ratio
        self.homophily = cfg.homophily
        self.sim_threshold = cfg.gnn.train.sim_threshold
        # ! Load data
        # data, num_classes = load_data(
        #    self.dataset_name, use_dgl=False, use_text=False, seed=self.seed)
        self.num_nodes = data.y.shape[0]
        self.num_classes = num_classes
        data.y = data.y.squeeze()
        # ! pick nodes for getting pseudo labels
        self.pl_mask, self.gold_mask = pick_nodes_random(data.train_mask, self.pl_rate, cfg.seed)
        self.pseudo_labels = torch.zeros_like(data.y) # TO BE UPDATED
        self.inconsistency_mask = torch.ones_like(data.y)
        # ! Init gnn feature
        topk = 3 if self.dataset_name == 'pubmed' else 5
        if self.feature_type == 'llm': # stored in folder embedding
            print("Loading features from llm embedding...")
            LLM_emb_path = f"llm_emb/{self.dataset_name}/{self.llm_model_name}.npy"
            print(f"LLM_emb_path: {LLM_emb_path}")
            features = data_normalization(torch.from_numpy(np.array(
                np.memmap(LLM_emb_path, mode='r',
                          dtype=np.float32,
                          shape=(self.num_nodes, 768))) #4096
            ).to(torch.float32))
        elif self.feature_type == 'ogb':
            print("Loading OGB features...")
            features = data.x
        elif self.feature_type == 're':
            print("Loading pretrained LM features (title and abstract) ...")
            LM_emb_path = f"/storage/qiaoyr/TAPE/prt_lm_1/{self.dataset_name}/{self.lm_model_name}-seed{self.seed}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, 768)))
            ).to(torch.float32)
        elif self.feature_type == 'TA':
            print("Loading pretrained LM features (title and abstract) ...")
            LM_emb_path = f"/storage/qiaoyr/TAPE/prt_lm/{self.dataset_name}/{self.lm_model_name}-seed{self.seed}.emb"
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
        print(self.features.shape)
        print(self.data)
        # ! Trainer init
        use_pred = self.feature_type == 'P'

        if self.gnn_model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif self.gnn_model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        elif self.gnn_model_name == "MLP":
            from core.GNNs.MLP.model import MLP as GNN
        elif self.gnn_model_name == "GAT":
            from core.GNNs.GAT.model import GAT as GNN
        elif self.gnn_model_name == "MixHop":
            from core.GNNs.MixHop.model import MixHop as GNN
        elif self.gnn_model_name == "APPNP":
            from core.GNNs.APPNP.model import APPNP as GNN
        elif self.gnn_model_name == "GCNII":
            from core.GNNs.GCNII.model import GCNII as GNN
        elif self.gnn_model_name == "GPRGNN":
            from core.GNNs.GPRGNN.model import GPRGNN as GNN
        elif self.gnn_model_name == "H2GCN":
            from core.GNNs.H2GCN.model import H2GCN as GNN
        elif self.gnn_model_name == "JKNet":
            from core.GNNs.JKNet.model import JKNet as GNN
        elif self.gnn_model_name == "SSP":
            from core.GNNs.SSP.model import SSP as GNN
        elif self.gnn_model_name == "SGC":
            from core.GNNs.SGC.model import SGC as GNN
        else:
            print(f"Model {self.gnn_model_name} is not supported! Loading MLP ...")
            from core.GNNs.MLP.model import MLP as GNN

        self.model = GNN(in_channels=self.features.shape[1],
                         hidden_channels=self.hidden_dim,
                         out_channels=self.num_classes,
                         num_layers=self.num_layers,
                         dropout=self.dropout).to(self.device)

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
        #print(logits)
        loss = self.loss_func(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        #print(loss)
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
    '''
    def update_pseudo_labels_and_features(self, pseudo_labels, emb): # pseudo_labels: list
        # update pseudo labels, noted that when pseudo label is -1, the corresponding node is not a pseudo label node
        org_pl_mask = self.pl_mask.clone()
        new_pseudo_labels = torch.zeros_like(self.data.y)
        new_pseudo_labels[org_pl_mask] = torch.tensor(pseudo_labels).to(self.device)
        new_emb = torch.zeros_like(self.features)
        new_emb[org_pl_mask] = emb.to(self.device)
        no_res_mask = (new_pseudo_labels == -1) # llm answer cant be resolved
        self.pl_mask = org_pl_mask & ~no_res_mask
        self.pseudo_labels = self.data.y.clone()
        self.pseudo_labels[self.pl_mask] = new_pseudo_labels[self.pl_mask]
        # get inconsistency mask
        self.inconsistency_mask = (self.data.y != self.pseudo_labels).to(self.device) # inconsistency of y and pseudo_labels
        # update features only when the pseudo label is consistent with the ground truth label
        self.features[~self.inconsistency_mask] = new_emb[~self.inconsistency_mask]
    '''
    
    def set_pl_mask(self, pl_mask):
        self.pl_mask = pl_mask.to(self.device)
        self.gold_mask = ~pl_mask.to(self.device)

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

        f_copy = self.features.clone()
   
        self.features[pl_nodes_list] = emb
        features_mask = (self.data.y != self.pseudo_labels).to(self.device) 
        self.features[features_mask] = f_copy[features_mask] 

    def update_pseudo_labels(self, pseudo_labels): # pseudo_labels: list
        
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
            #emb = torch.cat((emb[:index, :], emb[index + 1:, :]), dim=0)
        #emb = emb.to(self.device)
        #print('emb_shape:', emb.shape)
        print('orig_features_shape:', self.features.shape)
        print('pl_nodes_list_len:', len(pl_nodes_list))

        #f_copy = self.features.clone()
        #self.features[pl_nodes_list] = emb
        #features_mask = (self.data.y != self.pseudo_labels).to(self.device) 
        #self.features[features_mask] = f_copy[features_mask] 

    def augment_adjacency_matrix(self):
        # torch.manual_seed(42)
        inconsistency = (self.data.y != self.pseudo_labels) & self.pl_mask # inconsistency of y and pseudo_labels
        
        inconsistent_nodes_list = torch.nonzero(inconsistency).squeeze().tolist()
        print('inconsistent nodes num:', len(inconsistent_nodes_list))
        print('inconsistent nodes list:', inconsistent_nodes_list)
        edge_index = self.data.edge_index
        # delete all edges with inconsistent nodes
        mask = ~torch.any(edge_index[None, :] == torch.tensor(inconsistent_nodes_list).to(self.device)[:, None, None], dim=0)
        mask = torch.all(mask, dim=0)
        kept_edges = edge_index[:,mask] 
        for idx in inconsistent_nodes_list:
            # get the links involving this node from data.edge_index
            # data.edge_index is in the shape of (2, edges_num)
            mask = (edge_index[0] == idx) | (edge_index[1] == idx)
            selected_edges = edge_index[:,mask] #shape: (2, edges_num_of_idx)
            # cut off the edges involving this node according to cut_off_ratio
            remain_num = int(selected_edges.shape[1] * self.remain_ratio)
            edges_remain = selected_edges[:,torch.randperm(selected_edges.shape[1])[:(remain_num)]]
            # add edges_remain to kept_edges
            kept_edges = torch.cat((kept_edges, edges_remain), dim=1)
       
        #kept_edges = torch.unique(kept_edges, dim=1)
        print('augmented edges shape:', kept_edges.shape)
        self.data.edge_index = kept_edges

    def calculate_similarity_matrix(self):
        if self.homophily:
            fea_copy = self.features.clone().cpu().detach().numpy()
            sim_mat = cosine_similarity(X=fea_copy, Y=fea_copy)
        else:
            # for heterophilic graph, use graphlet degree vector to calculate sim # TODO
            fea_copy = self.features.clone().cpu().detach().numpy()
            sim_mat = cosine_similarity(X=fea_copy, Y=fea_copy)
        return torch.tensor(sim_mat).to(self.device)
        
    def augment_adjacency_matrix_sim(self):
        inconsistency = (self.data.y != self.pseudo_labels) & self.pl_mask
        inconsistent_nodes_list = torch.nonzero(inconsistency).squeeze().tolist()
        
        print('inconsistent nodes list:', inconsistent_nodes_list)
        edge_index = self.data.edge_index
        kept_edges = edge_index.clone()
        # delete all edges with inconsistent nodes
        mask = ~torch.any(edge_index[None, :] == torch.tensor(inconsistent_nodes_list).to(self.device)[:, None, None], dim=0)
        mask = torch.all(mask, dim=0)
        kept_edges = edge_index[:,mask] 
        print('kept_edges shape:', kept_edges.shape)
        fea_copy = self.features.clone().cpu().detach().numpy()
        sim_mat = cosine_similarity(fea_copy, fea_copy)

        for idx in inconsistent_nodes_list:
            #print('pruning edges of idx:', idx)
            mask = (edge_index[0] == idx) | (edge_index[1] == idx)
            selected_edges = edge_index[:, mask]
            #print('to be pruned edges shape:', selected_edges.shape)
            
            prune_index = []
            for i in range(selected_edges.shape[1]):
                source_idx = selected_edges[0, i]
                target_idx = selected_edges[1, i]
                if sim_mat[source_idx, target_idx] < self.sim_threshold:                                                            
                    prune_index.append(i)
            #print('prune_index:', prune_index)
            remain_mask = torch.ones(selected_edges.shape[1], dtype=torch.bool)
            remain_mask[prune_index] = False
            #print('remain_mask:', remain_mask)
            edges_remain = selected_edges[:, remain_mask]
            #print('edges_remain shape:', edges_remain.shape)
            kept_edges = torch.cat((kept_edges, edges_remain), dim=1)
    
        # Remove duplicate edges
        kept_edges, _ = kept_edges.unique(dim=1, return_inverse=True)
        
        print('original edges shape:', edge_index.shape)
        print('augmented edges shape:', kept_edges.shape)
        
        self.data.edge_index = kept_edges


        





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
