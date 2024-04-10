# %%
import numpy as np
import torch
import random
import torch_geometric.transforms as T
from torch_geometric.data import Data
import pandas as pd
import os
from pathlib import Path
import re



def parse_webkb(data_name):
    path = f'/storage/qiaoyr/TAPE/dataset/web_kb/WebKB/{data_name}'
    webpage_features_labels = np.genfromtxt("{}.content".format(path), dtype=np.dtype(str))
    data_X = webpage_features_labels[:, 1:-1].astype(np.float32)
    labels = webpage_features_labels[:, -1]
    #print(labels)
    class_map = {x: i for i, x in enumerate(['course', 'faculty', 'student','project', 'staff'])}  
    #print(class_map)
    data_Y = np.array([class_map[x] for x in labels])
    data_webpage_url = webpage_features_labels[:, 0]
    # data_webpage_id = np.arange(len(data_webpage_url))
    data_webpage_id_map = {x: i for i, x in enumerate(data_webpage_url)}
    print(path)
    edges_unordered = np.genfromtxt("{}.cites".format(path), dtype=np.dtype(str))
    '''
    for i in range(edges_unordered.shape[0]):
        if edges_unordered[i][0] == edges_unordered[i][1]:
            print('self loop:',edges_unordered[i][0])
    '''
    edges = np.array(list(map(data_webpage_id_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    #print(edges.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype=np.int32)
    #print(data_edges.shape)
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    #print(data_edges.shape)

    return data_X, data_Y, data_webpage_url, np.unique(data_edges, axis=0).transpose()
# %%
'''
X, Y, webpage_id, edges = parse_wisconsin()
print(X.shape)
print(Y.shape)
print(webpage_id.shape)
print(edges.shape)
'''
# %% \Data(x=x, edge_index=edge_index, y=y)
def get_webkb_casestudy(data_name, SEED=0):
    data_X, data_Y, data_webpage_url, data_edges = parse_webkb(data_name)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data = Data(x=torch.tensor(data_X).float(),
                 edge_index=torch.tensor(data_edges).long(), 
                 y=torch.tensor(data_Y).long(),
                 num_nodes=len(data_Y))

    
    return data, data_webpage_url
# %%
def html_process(input_string):
    # 使用正则表达式去掉所有 HTML 标签
    lines = input_string.split('\n')
    clean_text = ' '.join(lines[6:])

    #non_empty_lines = [line for line in clean_text if line.strip()]
    
    #tag_list = ['<.*?>', r'<ahref\s*=\s*".*?"\s*>', r'<a\shref\s*=\s*".*?"\s*>', r'<meta *.html>', r'<img src*>', r'<IMG SRC*">', r'<bodyBACKGROUND*>', r'<imgsrc*>', r'<AHREF*>', '\n']
    tag_list = ['<.*?>', '\n', r'<a\s+href\s*=\s*".*?"\s*>', r'<IMG\s+SRC\s*=\s*".*?"\s+ALT\s*=\s*".*?"\s*>']
    for tag in tag_list:
        clean_text = re.sub(tag, '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text
'''
def get_raw_text_webkb(data_name, use_text=False, seed=0):
    data, data_webpage_url = get_webkb_casestudy(data_name, seed)
    if not use_text:
        return data, None
    text = []
    clean_text = []
    category_list = ['course', 'faculty', 'student','project', 'staff']
    path = '/storage/qiaoyr/TAPE/dataset/web_kb_orig/webkb_raw'
    # print(data.y.shape)
    # for category in category_list:
        # webpages = os.listdir('{}/{}'.format(path, category))
    for i, url in enumerate(data_webpage_url):
        label = data.y[i]
        url = url.replace('/', '^')
        if not url.endswith('.html'):
            url += '^'
        try:
            file_path = '{}/{}/{}/{}'.format(path, category_list[label], data_name, url)
            t = open(file_path, 'r', errors='ignore').read()
            text.append(t)
        except:
            print(i, file_path, 'not found') ###TODO
            text.append('')
    for t in text:
        clean = html_process(t)
        clean_text.append(clean)
    return data, clean_text
'''
# %%
def delete_vacant_webpage(data, i):
    data.y = torch.cat((data.y[:i], data.y[(i+1):]))
    # data.edge_index = torch.cat((data.edge_index[:,:i], data.edge_index[:,(i+1):]), dim=1)
    data.x = torch.cat((data.x[:i], data.x[(i+1):]))
    data.num_nodes -= 1
    mask = (data.edge_index[0] == i) | (data.edge_index[1] == i)
    data.edge_index = data.edge_index[:,~mask] 
    return data
# %%
def get_raw_text_webkb(data_name, use_text=False, seed=0):
    data, data_webpage_url = get_webkb_casestudy(data_name, seed)

    text = []
    clean_text = []
    category_list = ['course', 'faculty', 'student','project', 'staff']
    path = '/storage/qiaoyr/TAPE/dataset/web_kb_orig/webkb_raw'
    # print(data.y.shape)
    # for category in category_list:
        # webpages = os.listdir('{}/{}'.format(path, category))
    for i, url in enumerate(data_webpage_url):
        label = data.y[i]
        url = url.replace('/', '^')
        pages_to_remove = []
        if not url.endswith('.html'):
            url += '^'
        file_path = '{}/{}/{}/{}'.format(path, category_list[label], data_name, url)
        if os.path.exists(file_path):
            t = open(file_path, 'r', errors='ignore').read()
            text.append(t)
        else:
            pages_to_remove.append(i)
            # print(i, file_path, 'not found') ###TODO
            # text.append('')
            
    if data_name == 'wisconsin':
        pages_to_remove = [3,5]
    elif data_name == 'cornell':
        pages_to_remove = [12]
    elif data_name == 'texas':
        pages_to_remove = [0]
    elif data_name == 'washington':
        pages_to_remove = [1, 152, 156,170,171,178,214,227]

    for i in reversed(pages_to_remove):
        data = delete_vacant_webpage(data, i)
    edge_index = data.edge_index
    out_of_range_edges = (edge_index[0] < 0) | (edge_index[0] >= data.num_nodes) | (edge_index[1] < 0) | (edge_index[1] >= data.num_nodes)

    data.edge_index = data.edge_index[:,~out_of_range_edges]
    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])
    if not use_text:
        return data, None
    for t in text:
        clean = html_process(t)
        clean_text.append(clean)
    return data, clean_text

# %%
data, clean_text = get_raw_text_webkb('wisconsin', use_text=True, seed=0)
# %%
print(data)
print(len(clean_text))


# %%
'''
path = '/storage/qiaoyr/TAPE/dataset/web_kb_orig/webkb_raw'
category_list = ['course', 'faculty', 'student','project', 'staff']
for category in category_list:
    file_path = '{}/{}/{}'.format(path, category, data_name)
    print(file_path)
    webpages = os.listdir(file_path)
    print(category, len(webpages))
'''
# %%for i in range(data.num_nodes):
'''
    mask = (data.edge_index[0] == i) | (data.edge_index[1] == i)
data.edge_index = data.edge_index[:,~mask] 
# %%
import torch
edge_index = data.edge_index
# 假设 edge_index 是一个大小为 (2, edge_num) 的张量
# 假设 n 是节点的数量

# 获取节点的数量 n
n = data.num_nodes  # 你需要将这个值替换为你实际的节点数量

# 检查 edge_index 中的边是否超出节点范围
out_of_range_edges = (edge_index[0] < 0) | (edge_index[0] >= n) | (edge_index[1] < 0) | (edge_index[1] >= n)

# 如果 out_of_range_edges 中存在 True 值，表示有边超出节点范围
if out_of_range_edges.any():
    print(out_of_range_edges)
    print("存在超出节点范围的边。")
else:
    print("所有边都在节点范围内。")



'''  

# %%
