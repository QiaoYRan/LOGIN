# %%
from core.utils import init_path
import numpy as np
import torch


def get_gnn_trainer(model):
    if model in ['GCN', 'RevGAT', 'SAGE']:
        from core.GNNs.gnn_trainer import GNNTrainer
    else:
        raise ValueError(f'GNN-Trainer for model {model} is not defined')
    return GNNTrainer


class Evaluator:
    def __init__(self, name):
        self.name = name

    def eval(self, input_dict):
        y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}


"""
Early stop modified from DGL implementation
"""


class EarlyStopping:
    def __init__(self, patience=10, path='es_checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        if isinstance(path, list):
            self.path = [init_path(p) for p in path]
        else:
            self.path = init_path(path)

    def step(self, acc, model, epoch):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        es_str = f'{self.counter:02d}/{self.patience:02d} | BestVal={self.best_score:.4f}@E{self.best_epoch}'
        return self.early_stop, es_str

    def save_checkpoint(self, model):
        """Saves model when validation loss decrease."""
        if isinstance(model, list):
            for i, m in enumerate(model):
                torch.save(m.state_dict(), self.path[i])
        else:
            torch.save(model.state_dict(), self.path)

# %%
import numpy as np
import torch
def pick_nodes_random(train_mask, pl_rate, seed=0):

    np.random.seed(seed)
    # 计算 train_mask 中 True 值的索引
    true_indices = torch.nonzero(train_mask).squeeze()  # 获取所有为 True 的索引

    # 获取 true_indices 的长度
    num_true_indices = len(true_indices)
    
    # 计算要抽取的数量（10%）
    num_samples = int(pl_rate * num_true_indices)

    # 使用随机采样来选择要抽取的索引
    if num_samples > 0:
        # 从 true_indices 中随机选择要抽取的索引
        sampled_indices = np.random.choice(true_indices.cpu().numpy(), size=num_samples, replace=False)
        
        # 创建一个新的 mask 来表示抽取的位置
        extracted_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        extracted_mask[sampled_indices] = True
        remaining_mask = train_mask.clone()
        remaining_mask[sampled_indices] = False  # 将已抽取的位置设为 False
        
        # 输出抽取的位置
        # print("抽取的位置：", extracted_mask)
    else:
        print("没有足够的 True 值来抽取样本。")

    return extracted_mask, remaining_mask


# %%
def data_normalization(org_data):
    d_min = org_data.min()
    if d_min < 0:
        org_data += torch.abs(d_min)
        d_min = org_data.min()
    d_max = org_data.max()
    dst = d_max - d_min
    norm_data = (org_data - d_min).true_divide(dst)
    return norm_data

