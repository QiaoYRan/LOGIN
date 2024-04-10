# %%
from load_pubmed import get_raw_text_pubmed
num_classes = 3
data, text = get_raw_text_pubmed(use_text=True, seed=1)

category_list = ["Diabetes Mellitus, Experimental", "Diabetes Mellitus Type 1", "Diabetes Mellitus Type 2"]

# %%
def distinguish_neighbors(idx, edge_index, nodes_id):
    neighbors_one_hop = []
    if edge_index.size(1) == 0:
        return [], []
    #for num in range(len(nodes_id)):
    for num in range(edge_index.size(1)):
        if edge_index[0][num] == idx:
            neighbors_one_hop.append(nodes_id[int(edge_index[1][num])])
        if edge_index[1][num] == idx:
            neighbors_one_hop.append(nodes_id[int(edge_index[0][num])])
    neighbors_one_hop = list(set(neighbors_one_hop))
    neighbors_two_hop = list(set(nodes_id) - set(neighbors_one_hop) - set([idx]))
    return neighbors_one_hop, neighbors_two_hop
# %%
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
import torch as th

pyg_data = Data(edge_index = data.edge_index, num_nodes = data.num_nodes, y=data.y)

def prompt_for_single_node(nidx):
    center_node = nidx 
    num_hops = 2
    num_neighbors = -1 # -1 means all neighbors

    # 邻居采样    
    sampler = NeighborLoader(pyg_data, input_nodes=th.Tensor([center_node]).long(),
                            num_neighbors=[num_neighbors] * num_hops, 
                            batch_size=1)

    # 获取子图     
    sampled_data = next(iter(sampler))
    neighbors_one_hop, neighbors_two_hops = distinguish_neighbors(nidx, sampled_data.edge_index, sampled_data.n_id.tolist())
#pertaining to diabetes classified into one of three classes
    # print(sampled_data.y.tolist())
    question = 'Question: Which diabetes class does this paper pertain to? Give the most likely diabetes class in this paper directly, choosing from ["Diabetes Mellitus, Experimental", "Diabetes Mellitus Type 1", "Diabetes Mellitus Type 2"].\n\
            Ensure that your response can be parsed by Python json, use the following format as an example: {"classification result": "Diabetes Mellitus, Experimental", "explanation": "your explanation for why this Diabetes here",}\n\
            Ensure that the classification result must match one of the given choices.}'
    
    graph_desc = "\nAnd in the 'node label' list / 'GNN-predicted node label'list, you'll find the pre-labeled / GNN-predicted subcategories corresponding to the neighbors within two hops of the target paper as per the 'node_list'.\n "
    dsname = 'pubmed'
    dict = {}
    dict['id'] = f'{dsname}_{nidx}'
    label_y = [category_list[i] for i in sampled_data.y.tolist()]
    dict['graph'] = {'node_idx': nidx, 'node_list': sampled_data.n_id.tolist(), 'one_hop_neighbors': neighbors_one_hop, 'two_hops_neighbors': neighbors_two_hops, 'node_label': label_y}
    # conv_list = []
    conv_temp = {}
    conv_temp['from'] = 'human'
    conv_temp['value'] = 'Given a pubmed citation graph: \n<graph>\n, where the 0th node is the target paper, with the following information: \n' + text[nidx] + graph_desc + question
    # conv_list.append(copy.deepcopy(conv_temp))
    dict['conversations'] = conv_temp
    return dict

# %%
import json
for node_idx in range(pyg_data.num_nodes):
    prompt = prompt_for_single_node(node_idx)
    print(prompt)
    file_path = f'/storage/qiaoyr/TAPE/prompts/pubmed/pubmed_{node_idx}.json'
    with open(file_path, 'w') as f:
        json.dump(prompt, f)
# %%
prompt = prompt_for_single_node(0)
print(prompt)