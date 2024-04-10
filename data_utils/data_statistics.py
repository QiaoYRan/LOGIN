# %%
import torch
def delete_reversed_edges(edge_index):
    # 将 edge_index 转换为元组，并排序每个元组
    sorted_edges = torch.sort(edge_index, dim=0).values

    # 使用 unique 去除重复的边
    unique_edge_index, _ = torch.unique(sorted_edges, dim=1, return_inverse=True)
    return unique_edge_index

# %%
import numpy as np
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops
from load import load_data

data, num_classes, text = load_data('cornell', use_dgl=False, use_text=True, seed=0)
print(data)
print(num_classes)
def node_homophily_edge_idx(edge_idx, labels, num_nodes):
    """ edge_idx is 2 x(number edges) """
    edge_index = remove_self_loops(edge_idx)[0]
    # edge_index = delete_reversed_edges(edge_index)
    # print(edge_index.shape)
    hs = torch.zeros(num_nodes)
    degs_0 = torch.bincount(edge_index[0,:]).float()
    degs = torch.bincount(edge_index[0,:]).float()
    #degs_1 = torch.bincount(edge_index[1,:]).float()
    #print(degs_0.shape, degs_1.shape)
    #degs = degs_0 + degs_1
    matches = (labels[edge_index[0,:]] == labels[edge_index[1,:]]).float()
    hs = hs.scatter_add(0, edge_index[0,:], matches) / degs
    #print('zero degree nodes num', torch.sum(degs == 0).item(), torch.sum(degs_1 == 0).item())
    # return hs[degs != 0].mean()
    return hs
def homo_stat(data):
    nh = node_homophily_edge_idx(data.edge_index, data.y, data.y.shape[0])
    print(nh.shape)
    threshold = 0.5
    count_below_threshold = torch.sum(nh < threshold).item()
    total_nodes = nh.shape[0]
    percentage_below_threshold = (count_below_threshold / total_nodes) * 100.0
    print(nh)
    print(f"heter 的点的比例为: {percentage_below_threshold}%")
homo_stat(data)
# %%
def homophily_node(data):



    edge_idx = data.edge_index
    Y = data.y
    edge_homo = (Y[edge_idx[0]] == Y[edge_idx[1]]).float()
    node_edges_num = torch.sum(Y[edge_idx[0]] == Y[edge_idx[1]])

    edge_to_label = (Y[edge_idx[0]] == Y[edge_idx[1]]).float()
    print(edge_to_label.shape)
    node_probabilities = torch.mean(edge_to_label, dim=1)
    # 假设 node_probabilities 包含了每个节点的概率
    threshold = 0.5
    count_below_threshold = torch.sum(node_probabilities < threshold).item()
    total_nodes = len(node_probabilities)
    percentage_below_threshold = (count_below_threshold / total_nodes) * 100.0

    print(f"heter 的点的比例为: {percentage_below_threshold}%")
homophily_node(data)
# %%
def edge_homophily(A, labels, ignore_negative=False):
    """ gives edge homophily, i.e. proportion of edges that are intra-class
    compute homophily of classes in labels vector
    See Zhu et al. 2020 "Beyond Homophily ..."
    if ignore_negative = True, then only compute for edges where nodes both have
        nonnegative class labels (negative class labels are treated as missing
    *** intra-class edges/total edges
    """
    src_node, targ_node = A.nonzero()
    matching = labels[src_node] == labels[targ_node]
    labeled_mask = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    if ignore_negative:
        edge_hom = np.mean(matching[labeled_mask])
    else:
        edge_hom = np.mean(matching)
    return edge_hom

def compat_matrix(A, labels):
    """ c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j 
     of edges incident to class i nodes 
     See Zhu et al. 2020
    """
    c = len(np.unique(labels))
    H = np.zeros((c,c))
    src_node, targ_node = A.nonzero()
    for i in range(len(src_node)):
        src_label = labels[src_node[i]]
        targ_label = labels[targ_node[i]]
        H[src_label, targ_label] += 1
    H = H / np.sum(H, axis=1, keepdims=True)
    return H

def node_homophily(A, labels):
    """ average of homophily for each node
    """
    src_node, targ_node = A.nonzero()
    edge_idx = torch.tensor(np.vstack((src_node, targ_node)), dtype=torch.long).contiguous()
    labels = torch.tensor(labels)
    num_nodes = A.shape[0]
    return node_homophily_edge_idx(edge_idx, labels, num_nodes)

def edge_homophily_edge_idx(edge_idx, labels):
    """ edge_idx is 2x(number edges) """
    edge_index = remove_self_loops(edge_idx)[0]
    return torch.mean((labels[edge_index[0,:]] == labels[edge_index[1,:]]).float())

# %%
def node_homophily_edge_idx(edge_idx, labels, num_nodes):
    """ edge_idx is 2 x(number edges) """
    edge_index = remove_self_loops(edge_idx)[0]
    hs = torch.zeros(num_nodes)
    degs = torch.bincount(edge_index[0,:]).float()
    matches = (labels[edge_index[0,:]] == labels[edge_index[1,:]]).float()
    hs = hs.scatter_add(0, edge_index[0,:], matches) / degs
    # return hs[degs != 0].mean()
    return hs
# %%  
def compat_matrix_edge_idx(edge_idx, labels):
    """
     c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j 
     of edges incident to class i nodes 
     "Generalizing GNNs Beyond Homophily"
     treats negative labels as unlabeled
     """
    edge_index = remove_self_loops(edge_idx)[0]
    src_node, targ_node = edge_index[0,:], edge_index[1,:]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze()
    c = label.max()+1
    H = torch.zeros((c,c)).to(edge_index.device)
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    label_idx = torch.cat((src_label.unsqueeze(0), targ_label.unsqueeze(0)), axis=0)
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        scatter_add(torch.ones_like(add_idx).to(H.dtype), add_idx, out=H[k,:], dim=-1)
    H = H / torch.sum(H, axis=1, keepdims=True)
    return H

def our_measure(edge_index, label):
    """ 
    our measure \hat{h}
    treats negative labels as unlabeled 
    """
    label = label.squeeze()
    c = label.max()+1
    H = compat_matrix_edge_idx(edge_index, label)
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1]
    proportions = counts.float() / nonzero_label.shape[0]
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k,k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c-1
    return val

