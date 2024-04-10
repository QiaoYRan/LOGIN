# %%
from load_arxiv import get_raw_text_arxiv as get_raw_text
num_classes = 40
data, text = get_raw_text(use_text=True, seed=1)
# %%
print(data.train_mask)
print(data.y)
# %%
def pick_nodes_random(train_mask, pl_rate):
    import torch
    import numpy as np

    # 计算 train_mask 中 True 值的索引
    true_indices = torch.nonzero(train_mask).squeeze()  # 获取所有为 True 的索引

    # 获取 true_indices 的长度
    num_true_indices = len(true_indices)
    
    # 计算要抽取的数量（10%）
    num_samples = int(pl_rate * num_true_indices)

    # 使用随机采样来选择要抽取的索引
    if num_samples > 0:
        # 从 true_indices 中随机选择要抽取的索引
        sampled_indices = np.random.choice(true_indices.numpy(), size=num_samples, replace=False)
        
        # 创建一个新的 mask 来表示抽取的位置
        extracted_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        extracted_mask[sampled_indices] = True
        
        # 输出抽取的位置
        print("抽取的位置：", extracted_mask)
        remaining_mask = train_mask.clone()
        remaining_mask[sampled_indices] = False  # 将已抽取的位置设为 False
        print("剩余的位置：", remaining_mask)
    else:
        print("没有足够的 True 值来抽取样本。")

    return extracted_mask, remaining_mask
# %%
import torch
pl_mask, gold_mask = pick_nodes_random(data.train_mask, 0.1)
print(len(torch.nonzero(pl_mask).squeeze()))
true_indices = torch.nonzero(data.train_mask).squeeze() 
print(len(true_indices))
print(len(torch.nonzero(gold_mask).squeeze()))
# %%
import pandas as pd

nodeidx2paperid = pd.read_csv(
    'dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')
nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].astype(int)
raw_text = pd.read_csv('/storage/qiaoyr/TAPE/dataset/ogbn_arxiv_orig/titleabs.tsv',
                        sep='\t', header=None, names=['paper id', 'title', 'abs'])
raw_text['paper id'] = raw_text['paper id'].astype(int)
df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
text = []
for ti, ab in zip(df['title'], df['abs']):
    t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
    text.append(t)
# %%
print(raw_text)
# %%
print(nodeidx2paperid)
# %%
df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
# %%
print(data)

# %%
# %%
# try cora
from load_cora import get_raw_text_cora 
num_classes = 7
data, text = get_raw_text(use_text=True, seed=1)
# %%
from load_cora import get_raw_text_cora
data, text = get_raw_text_cora(use_text=True, seed=1)
# %%
print(data.adj_t)
# %%
from load import load_data

data, num_classes, text = load_data('wisconsin', use_dgl=False, use_text=True, seed=0)