# %%
from load_arxiv_2023 import get_raw_text_arxiv_2023
import pandas as pd
num_classes = 40
data, text = get_raw_text_arxiv_2023(use_text=True, seed=1)
arxiv_category = pd.read_csv('/storage/qiaoyr/TAPE/dataset/ogbn_arxiv/mapping/augmented_labelidx2arxivcategeory.csv',
                        sep=',')
category_list = []
for category in arxiv_category['arxiv category']:
    category_list.append(category)
print(len(category_list))
# %%
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
def trim_string(text, num_words_to_remove):
    words = text.split()
    
    if len(words) > num_words_to_remove:
        trimmed_words = words[:-num_words_to_remove]
        trimmed_text = ' '.join(trimmed_words)
        return trimmed_text
    else:
        return "字符串太短，无法减少指定数量的单词"
        
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
    question = 'Question: Which cs subcategory does this paper belong to? Give the most likely subcategory of this paper directly, choosing from  ["Numerical Analysis", "Multimedia", "Logic in Computer Science", "Computers and Society", "Cryptography and Security", "Distributed, Parallel, and Cluster Computing", "Human-Computer Interaction", "Computational Engineering, Finance, and Science", "Networking and Internet Architecture", "Computational Complexity", "Artificial Intelligence", "Multiagent Systems", "General Literature", "Neural and Evolutionary Computing", "Symbolic Computation", "Hardware Architecture", "Computer Vision and Pattern Recognition", "Graphics", "Emerging Technologies", "Systems and Control", "Computational Geometry", "Other Computer Science", "Programming Languages", "Software Engineering", "Machine Learning", "Sound", "Social and Information Networks", "Robotics", "Information Theory", "Performance", "Computation and Language", "Information Retrieval", "Mathematical Software", "Formal Languages and Automata Theory", "Data Structures and Algorithms", "Operating Systems", "Computer Science and Game Theory", "Databases", "Digital Libraries", "Discrete Mathematics"]\
            Ensure that your response can be parsed by Python json, use the following format as an example: {"classification result": "Numerical Analysis", "explanation": "your explanation for your classification here",}\n\
                                                                                                             Ensure that the classification result must match one of the given choices.}'
    
    graph_desc = "\nAnd in the 'node label' list / 'GNN-predicted node label'list, you'll find the pre-labeled / GNN-predicted subcategories corresponding to the neighbors within two hops of the target paper as per the 'node_list'.\n "
    dsname = 'arxiv_2023'
    dict = {}
    dict['id'] = f'{dsname}_{nidx}'
    label_y = [category_list[i] for i in sampled_data.y.tolist()]
    dict['graph'] = {'node_idx': nidx, 'node_list': sampled_data.n_id.tolist(), 'one_hop_neighbors': neighbors_one_hop, 'two_hops_neighbors': neighbors_two_hops, 'node_label': label_y}
    # conv_list = []
    conv_temp = {}
    conv_temp['from'] = 'human'
    #t = trim_string(text[nidx], 90)
    t = text[nidx]
    conv_temp['value'] = 'Given a citation graph: \n<graph>\n, where the 0th node is the target paper, with the following information: \n' + t + graph_desc + question
    # conv_list.append(copy.deepcopy(conv_temp))
    dict['conversations'] = conv_temp
    return dict

# %%
import json
for node_idx in range(pyg_data.num_nodes):
    prompt = prompt_for_single_node(node_idx)
    print(prompt)
    file_path = f'/storage/qiaoyr/TAPE/prompts/arxiv_2023/arxiv_2023_{node_idx}.json'
    with open(file_path, 'w') as f:
        json.dump(prompt, f)
# %%
'''
prompt = prompt_for_single_node(0)
print(prompt)
# %%
def trim_string(text, num_words_to_remove):
    words = text.split()
    
    if len(words) > num_words_to_remove:
        trimmed_words = words[:-num_words_to_remove]
        trimmed_text = ' '.join(trimmed_words)
        return trimmed_text
    else:
        return "字符串太短，无法减少指定数量的单词"
# %%
# 要处理的字符串
input_string = "ads vdsf dsg dsdg gdsth sgfjdh rhtsrag thd。"
# 要减少的单词数量
num_words_to_remove = 5

# 调用函数进行减少单词操作
result_string = trim_string(input_string, num_words_to_remove)

# 打印结果
print(result_string)
# %%
'''