# %%
from load_arxiv import get_raw_text_arxiv
import pandas as pd
num_classes = 40
data, text = get_raw_text_arxiv(use_text=True, seed=1)
arxiv_category = pd.read_csv('/storage/qiaoyr/TAPE/dataset/ogbn_arxiv/mapping/augmented_labelidx2arxivcategeory.csv',
                        sep=',')
category_list = []
for category in arxiv_category['arxiv category']:
    category_list.append(category)

# %%
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data

s = data.adj_t.to_scipy() # 转换为稀疏张量
edge_index, edge_attr = from_scipy_sparse_matrix(s) # 转换为COO格式
pyg_data = Data(edge_index = edge_index, edge_attr = edge_attr, num_nodes = data.num_nodes, y=data.y)

# %%
import torch as th

def prompt_for_single_node(nidx):
    center_node = nidx 
    num_hops = 2
    num_neighbors = 10

    # 邻居采样    
    sampler = NeighborLoader(pyg_data, input_nodes=th.Tensor([center_node]).long(),
                            num_neighbors=[num_neighbors] * num_hops, 
                            batch_size=1)

    # 获取子图     
    sampled_data = next(iter(sampler))

    question = 'Question: Which arXiv CS sub-category does this paper belong to? Give the most likely arXiv CS sub-categories of this paper directly, in the form "cs.XX" with full name of the category.'
    graph_desc = "\nAnd in the 'node label' list, you'll find subcategories corresponding to the 2-hop neighbors of the target paper as per the 'node_list'."
    dsname = 'arxiv'
    dict = {}
    dict['id'] = f'{dsname}_{nidx}'
    label_y = [category_list[i[0]] for i in sampled_data.y.tolist()]
    dict['graph'] = {'node_idx':nidx, 'edge_index': sampled_data.edge_index.tolist(), 'node_list': sampled_data.n_id.tolist(), 'node_label': label_y}
    # conv_list = []
    conv_temp = {}
    conv_temp['from'] = 'human'
    conv_temp['value'] = 'Given a citation graph: \n<graph>\n, where the 0th node is the target paper, with the following information: \n' + text[nidx] + graph_desc + question
    # conv_list.append(copy.deepcopy(conv_temp))
    dict['conversations'] = conv_temp
    return dict

# %%
import json
for node_idx in range(pyg_data.num_nodes):
    prompt = prompt_for_single_node(node_idx)
    print(prompt)
    file_path = f'/storage/qiaoyr/TAPE/prompts/ogbn_arxiv/arxiv_{node_idx}.json'
    with open(file_path, 'w') as f:
        json.dump(prompt, f)

# %%
import torch as th
import copy
nidx = 0
for nidx in range(pyg_data.num_nodes):
    file_starting_num = nidx % 10000

    center_node = nidx 
    num_hops = 2
    num_neighbors = 10

    # 邻居采样    
    sampler = NeighborLoader(pyg_data, input_nodes=th.Tensor([center_node]).long(),
                            num_neighbors=[num_neighbors] * num_hops, 
                            batch_size=1)

    # 获取子图     
    sampled_data = next(iter(sampler))

    question = 'Question: Which arXiv CS sub-category does this paper belong to? Give the most likely arXiv CS sub-categories of this paper directly, in the form "cs.XX" with full name of the category.'
    dsname = 'arxiv'
    dict = {}
    dict['id'] = f'{dsname}_{nidx}'
    label_y = [category_list[i[0]] for i in sampled_data.y.tolist()]
    dict['graph'] = {'node_idx':nidx, 'edge_index': sampled_data.edge_index.tolist(), 'node_list': sampled_data.n_id.tolist(), 'node_label': label_y}
    # conv_list = []
    conv_temp = {}
    conv_temp['from'] = 'human'
    conv_temp['value'] = 'Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, with the following information: \n' + text[nidx] + question
    # conv_list.append(copy.deepcopy(conv_temp))
    dict['conversations'] = conv_temp
    print(dict)




# %%


