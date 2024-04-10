
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
        elif self.feature_type == 'TA':
            print("Loading pretrained LM features (title and abstract) ...")
            LM_emb_path = f"/storage/qiaoyr/TAPE/prt_lm/{self.dataset_name}/{self.lm_model_name}-seed{self.seed}.emb" # TODO set seed 0
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
                            dropout=self.dropout
                         ).to(self.device)

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

    def _pretrain_adj(self, adj):
        # ! Shared
        self.model.train()
        self.optimizer.zero_grad()
        # ! Specific
        logits = self._forward(self.features, adj)
        #print(logits)
        loss = self.loss_func(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        #print(loss)
        train_acc = self.evaluator(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_acc, logits[self.data.train_mask]
    '''
    def _cal_adj(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes)).to(self.device)
        for i in range(self.data.edge_index.shape[1]):
            source_node = self.data.edge_index[0, i].item()
            target_node = self.data.edge_index[1, i].item()
            adj[source_node, target_node] = 1
            adj[target_node, source_node] = 1
        return adj
    '''



    def _cal_adj(self):
        edge_index = self.data.edge_index
        num_nodes = self.num_nodes

        # 合并 row 和 col 张量
        indices = torch.stack([edge_index[0], edge_index[1]], dim=0)

        # 使用稀疏张量创建邻接矩阵，并指定数据类型为 float32
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0], dtype=torch.float32), (num_nodes, num_nodes))

        # 将稀疏张量转换为密集张量（如果需要）
        #adj = adj.to_dense()

        return adj

    
    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self._forward(self.features, self.data.edge_index)
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, logits

    
    @ torch.no_grad()
    def _evaluate_adj(self, adj):
        self.model.eval()
        logits = self._forward(self.features, adj)
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, logits
    
    def set_pl_mask(self, pl_mask):
        self.pl_mask = pl_mask
        self.gold_mask = ~pl_mask

    

    @time_logger
    def train(self, prt_sign=True):
        # ! Training
        pseudo_labels = self.pseudo_labels
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            if self.gnn_model_name == 'GCNII':
                adj = self._cal_adj()
                loss, train_acc, logits = self._pretrain_adj(adj)
                val_acc, test_acc, _ = self._evaluate_adj(adj)
            else:
                loss, train_acc, logits = self._pretrain()
                val_acc, test_acc, _ = self._evaluate()
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
        if self.gnn_model_name == 'GCNII':
            adj = self._cal_adj()
            val_acc, test_acc, logits = self._evaluate_adj(adj)
        else:
            val_acc, test_acc, logits = self._evaluate()
        print(val_acc, test_acc)
        print(
            f'[{self.gnn_model_name} + {self.feature_type}] ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        return logits, res
