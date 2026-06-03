import os
import numpy as np
import time
import datetime
import pytz
import torch

def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


# * ============================= Time Related =============================


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper

def calculate_uncertainty_score(logits_list):
    # calculate uncertainty score
    # logits_list: list of logits from different GNNs
    # return: uncertainty score
    variance = torch.var(torch.stack(logits_list), dim=0)
    uncertainty_score = torch.sum(variance, dim=1)

    return uncertainty_score

def find_topk_uncertain_nodes(uc_score, pl_rate):
    # uc_score: uncertainty score of all nodes, tensor with the shape (nodes_num)
    # return: topk uncertain nodes' indices

    # 计算不确定度最高的10%的点的数量
    topk_percent = pl_rate
    k = int(topk_percent * uc_score.size(0))

    # 使用 torch.topk 找到不确定度最高的点的索引
    topk_values, topk_indices = torch.topk(uc_score, k)

    # 创建布尔掩码（pl_mask）
    pl_mask = torch.zeros_like(uc_score, dtype=torch.bool)
    pl_mask[topk_indices] = True

    return pl_mask

def find_misclassified_nodes(logits, labels):
    # logits: logits from GNN, tensor with the shape (nodes_num, num_classes)
    # labels: ground truth labels, tensor with the shape (nodes_num)
    # return: misclassified nodes' indices

    predicted_labels = torch.argmax(logits, dim=1)

    # 找到被分错的点的索引
    misclassified_indices = (predicted_labels != labels).nonzero().view(-1)

    # 输出被分错的点的列表
    print("Misclassified indices:", misclassified_indices)

    return misclassified_indices

def ensemble_logits(pre_logits, logits, pl_mask):
    # pre_logits: logits from pre-trained GNN, tensor with the shape (nodes_num, num_classes)
    # logits: logits from GNN, tensor with the shape (nodes_num, num_classes)
    # pl_mask: pseudo label mask, tensor with the shape (nodes_num)
    # return: ensemble logits
    new_logits = torch.where(pl_mask.unsqueeze(1), logits, pre_logits)
    return new_logits


def evaluate_with_logits(logits, trainer):
    # logits: logits from GNN, tensor with the shape (nodes_num, num_classes)
    # trainer: trainer object
    # return: train/val/test accuracy
    train_acc = trainer.evaluator(logits[trainer.data.train_mask], trainer.data.y[trainer.data.train_mask])
    val_acc = trainer.evaluator(logits[trainer.data.val_mask], trainer.data.y[trainer.data.val_mask])
    test_acc = trainer.evaluator(logits[trainer.data.test_mask], trainer.data.y[trainer.data.test_mask])
    return {'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc}

 
def process_pseudo_labels_and_emb(pseudo_labels, emb, pl_mask):
    # pseudo_labels: pseudo labels, list with the length (selected_nodes_num)
    # emb: embeddings of explanations, tensor with the shape (selected_nodes_num, emb_dim)
    # pl_mask: pseudo label mask, tensor with the shape (nodes_num)
    # return: pseudo_labels with the shape (nodes_num), emb with the shape (nodes_num, emb_dim)

    new_pseudo_labels = torch.zeros_like(pl_mask)
    new_pseudo_labels[pl_mask] = torch.tensor(pseudo_labels)
    new_emb = torch.zeros_like(emb)
    new_emb[pl_mask] = emb
    no_res_mask = (new_pseudo_labels == -1)
    pl_mask = pl_mask & ~no_res_mask
    return new_pseudo_labels, new_emb

