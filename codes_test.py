
# %%
import torch

# 假设有模型的输出 logits 和真实标签 y_label
logits = torch.randn(2708, 7)  # 2708个点，7个类别
y_label = torch.randint(0, 7, (2708,))

# 获取模型预测的类别
predicted_labels = torch.argmax(logits, dim=1)

# 找到被分错的点的索引
misclassified_indices = (predicted_labels != y_label).nonzero().view(-1)

# 输出被分错的点的列表
print("Misclassified indices:", misclassified_indices)

# %%
print(y_label.shape)
# %%
import torch

def ensemble_logits(pre_logits, logits, pl_mask):
    # 对于 pl_mask 为 True 的位置，使用 logits；否则，使用 pre_logits
    new_logits = torch.where(pl_mask.unsqueeze(1), logits, pre_logits)
    
    return new_logits

# 示例使用
nodes_num = 2708
num_classes = 10

# 假设有预训练的 logits、当前模型的 logits 和伪标签 mask
pre_logits = torch.randn(nodes_num, num_classes)
logits = torch.randn(nodes_num, num_classes)
pl_mask = torch.randint(0, 2, (nodes_num,), dtype=torch.bool)

# 使用 ensemble_logits 函数
ensemble_result = ensemble_logits(pre_logits, logits, pl_mask)

# 打印结果
print(ensemble_result.shape)  # 应该是 (nodes_num, num_classes)

# %%
print(pl_mask)
# %%
import torch
import random

# Number of nodes in the graph
num_nodes = 10  # Adjust this based on your graph size

# Number of edges you want to generate
num_edges = 20  # Adjust this as needed

# Generate random edges
edges = [(random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)) for _ in range(num_edges)]

# Convert the list of edges to a PyTorch tensor
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Ensure that edge_index is in the format (2, num_edges)
if edge_index.size(0) != 2:
    edge_index = edge_index.view(2, -1)

# Print the generated edge_index
print("Generated edge_index:")
print(edge_index)

import torch

# Assuming you have the node index you want to find edges for in 'i'
i = 3  # Replace this with the node index you are interested in

# Assuming 'edge_index' is your edge index tensor
# edge_index is in the size of (edges_num, 2)

# Find all the edges that include node 'i'
edges_with_i = torch.where((edge_index == i).any(dim=1))

# The 'edges_with_i' tensor now contains the indices of edges that include node 'i'
print(edges_with_i)
# %%
import math
mask = (edge_index[0] == 1) | (edge_index[1] == 1)
selected_edges = edge_index[:,mask]# %%
# cut off the edges involving this node according to cut_off_ratio
cut_off_num = math.ceil(selected_edges.shape[1] * 0.5)
# use random
edges_to_cut = selected_edges[:,torch.randperm(selected_edges.shape[1])[:cut_off_num]]

# %%
print(edge_index)
print(mask)
print(selected_edges)
print(edges_to_cut)
# %%
print(edge_index[:,~mask])
# %%
import torch

# 假设你有一个 edge_index tensor
edge_index = torch.tensor([[0, 1, 1, 2, 3, 2],
                           [1, 0, 2, 1, 1, 3]])

# 要删除的节点列表
nodes_to_remove = [2, 3]

# 生成一个掩码，标记需要保留的边
mask = ~torch.any(edge_index[None, :] == torch.tensor(nodes_to_remove)[:, None, None], dim=0)
print(mask)
mask = torch.all(mask, dim=0)
print(mask)
# 使用掩码删除不需要的边
filtered_edge_index = edge_index[:, mask]

# 打印结果
print("筛选后的边:")
print(filtered_edge_index)

# %%
import math
y = torch.tensor([0, 1, 2, 3, 4])
pseudo_labels = torch.tensor([0, 1, 4, 4, 4])
cut_off_ratio = 0.5
edge_index = torch.tensor([[0, 1, 1, 2, 3, 2, 3],
                           [1, 0, 2, 1, 1, 3, 4]])
inconsistency = (y != pseudo_labels) # inconsistency of y and pseudo_labels
inconsistent_nodes_list = torch.nonzero(inconsistency).squeeze().tolist()
print('inconsistent nodes num:', len(inconsistent_nodes_list))
print('inconsistent nodes list:', inconsistent_nodes_list)


mask = ~torch.any(edge_index[None, :] == torch.tensor(inconsistent_nodes_list)[:, None, None], dim=0)
mask = torch.all(mask, dim=0)
kept_edges = edge_index[:,mask] 
print(kept_edges)
for idx in inconsistent_nodes_list:
    # get the links involving this node from data.edge_index
    # data.edge_index is in the shape of (2, edges_num)
    mask = (edge_index[0] == idx) | (edge_index[1] == idx)
    selected_edges = edge_index[:,mask] #shape: (2, edges_num_of_idx)
    # cut off the edges involving this node according to cut_off_ratio
    cut_off_num = math.ceil(selected_edges.shape[1] * cut_off_ratio)
    print(cut_off_num)
    edges_not_to_cut = selected_edges[:,torch.randperm(selected_edges.shape[1])[:(selected_edges.shape[1]-cut_off_num)]]
    # add edges_not_to_cut to kept_edges
    print(edges_not_to_cut)
    kept_edges = torch.cat((kept_edges, edges_not_to_cut), dim=1)
    print(kept_edges)
# %%
import torch
y = torch.tensor([0, 1, 2, 3, 4])
pseudo_labels = torch.tensor([0, 0, 2, 3, 5])
d = torch.tensor([6, 7, 8, 9, 10])
mask = (y != pseudo_labels)
print(mask)
y[mask] = d[mask]
print(y)
# %%
import torch
y = torch.tensor([[0, 0, 2, 3, 4],[0, 0, 2, 3, 5]])
print(y.shape)
# %%
print(torch.unique(y,dim=1))
# %%
import os
import sys
sys.path.append(os.path.abspath('..'))
from core.data_utils.load import load_data, load_gpt_preds
data, num_classes, text = load_data('wisconsin', use_dgl=False, use_text=True, seed=0)
edge_index = data.edge_index
idx = 3
mask = (edge_index[0] == idx) | (edge_index[1] == idx)
selected_edges = edge_index[:,mask] #shape: (2, edges_num_of_idx)
print('to be pruned edges shape:', selected_edges.shape)
# %%
print(text[0])
# %%
import numpy as np

# 创建三个示例NumPy数组
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
array3 = np.array([7, 8, 9])

# 将这些数组保存到一个文件中
np.savez('/storage/qiaoyr/TAPE/my_arrays.npz', array1=array1, array2=array2, array3=array3)

# 载入文件并访问保存的数组
loaded_data = np.load('my_arrays.npz')
loaded_array1 = loaded_data['array1']
loaded_array2 = loaded_data['array2']
loaded_array3 = loaded_data['array3']

# 打印加载的数组
print("Loaded Array 1:", loaded_array1)
print("Loaded Array 2:", loaded_array2)
print("Loaded Array 3:", loaded_array3)

# %%
def trim_string(text, num_words_to_remove):
    words = text.split()
    
    if len(words) > num_words_to_remove:
        trimmed_words = words[:-num_words_to_remove]
        trimmed_text = ' '.join(trimmed_words)
        return trimmed_text
    else:
        return "字符串太短，无法减少指定数量的单词"

# 要处理的字符串
input_string = "这是一个示例字符串，可以根据需要减少其结尾的单词数量。"
# 要减少的单词数量
num_words_to_remove = 5

# 调用函数进行减少单词操作
result_string = trim_string(input_string, num_words_to_remove)

# 打印结果
print(result_string)

# %%
