# %%
import torch
import torch.nn.functional as F
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import re
import os
import sys
sys.path.append(os.path.abspath('/storage/qiaoyr/TAPE'))
from core.LLMs.conversation import conv_templates
import ray
from core.GNNs.gnn_utils import pick_nodes_random
from core.data_utils.load import load_data
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
os.environ["RAY_DEDUP_LOGS"] = "0"
LLM_model = 'lmsys/vicuna-7b-v1.5'
@torch.inference_mode()

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def prompt_cora_per_node(node_idx, y_pred):
    category = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    file_path = f'/storage/qiaoyr/TAPE/prompts/cora/cora_{node_idx}.json'
    with open(file_path, 'r') as f:
        prompt = json.load(f)
    neighbor_node_list = prompt['graph']['node_list'] 
    neighbor_node_list = [int(node) for node in neighbor_node_list]
    GNN_predicted_label_list = [y_pred[i] for i in neighbor_node_list]
    GNN_predicted_label_list = [category[i] for i in GNN_predicted_label_list]
    prompt['graph']['GNN_predicted_node_label'] = GNN_predicted_label_list
    return prompt


# %%
@torch.inference_mode()
def prompt_cora_parallel(temperature, pl_mask, logits, num_gpus):
    # get batch split up
    pl_nodes_list = torch.nonzero(pl_mask).squeeze().tolist()  
    nodes_num = len(pl_nodes_list)
    chunk_size = nodes_num // num_gpus
    ans_handles = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = nodes_num if i == num_gpus - 1 else (i + 1) * chunk_size
        batch_nodes = pl_nodes_list[start:end]
        ans_handles.append(prompt_cora_batch.remote(temperature, logits, batch_nodes))

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))
    # return ans_jsons

# %%

@ray.remote(num_gpus=1)
@torch.inference_mode()
def prompt_cora_batch(temperature, logits, nodes):
    '''
    input: temperature, gnn_predicted_labels(logits), nodes in this batch
    '''
    # get GNN predicted labels
    import os
    import sys
    sys.path.append(os.path.abspath('/storage/qiaoyr/TAPE'))
    probabilities = F.softmax(logits, dim=1)
    y_pred = torch.argmax(probabilities, dim=1)
    y_pred = y_pred.detach().cpu().numpy()


    disable_torch_init()
    print('start loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(LLM_model)
    print('finish loading')

    print('start loading LM')
    model = AutoModelForCausalLM.from_pretrained(LLM_model, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    for idx in tqdm(nodes):
        prompt = prompt_cora_per_node(idx, y_pred)
        qs = prompt['conversations']['value']
        pattern = r'<graph>' 
        qs = re.sub(pattern, str(prompt['graph']), qs)
        conv_mode = "vicuna_v1_1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()
        # print(idx)
        # print('prompt:', final_prompt)
        with torch.no_grad():
            input_ids = tokenizer([final_prompt], max_length=10000, truncation=True).input_ids
            output_ids = model.generate(
                torch.as_tensor(input_ids).cuda(),
                do_sample=True,
                temperature=temperature,
                max_new_tokens=1500, # 1024 before
            )
            output_ids = output_ids[0][len(input_ids[0]) :]
            outputs = tokenizer.decode(output_ids, skip_special_tokens=True, truncation=True).strip()
        # print('answer:', outputs)
        res_data = {"id": prompt["id"], "node_idx": prompt["graph"]["node_idx"], "res": outputs}.copy()
        res_file = f'/storage/qiaoyr/TAPE/prompts/LLMs/vicuna/cora_test/cora_{idx}.json'
        with open(res_file, 'w') as f:
            json.dump(res_data, f)
    return idx


# %%

def answer_parser_cora_per_node(idx):
    category_list = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    file_path = f'/storage/qiaoyr/TAPE/prompts/LLMs/vicuna/cora_test/cora_{idx}.json'
    with open(file_path, 'r') as f:
        answer = json.load(f)
    # print(answer)
    # print(answer['res'])
    try:
        dict_ans = eval(answer['res'])
        classification_result = dict_ans['classification result']
        label = category_list.index(classification_result)
        # print(label)
    except:
        print(f'cant resolve classification_result for idx {idx}')
        label = -1
    return label

def answer_parser_cora(pl_mask):
    pl_nodes_list = torch.nonzero(pl_mask).squeeze().tolist()  
    pseudo_labels = []
    for i in pl_nodes_list:
        pseudo_labels.append(answer_parser_cora_per_node(i))
    pseudo_labels = torch.tensor(pseudo_labels)
    return pseudo_labels

# %%
def prompt_cora(temperature, pl_mask, logits, gpu_nums):
    # data, num_classes = load_data('cora', use_dgl=False, use_text=False, seed=0)
    ray.init(num_gpus=gpu_nums)
    with torch.no_grad():
        prompt_cora_parallel(temperature, pl_mask, logits, gpu_nums)
    ray.shutdown()
    pl = answer_parser_cora(pl_mask)
    print(pl)
    return pl

if __name__ == '__main__':
    data, num_classes = load_data(
                'cora', use_dgl=False, use_text=False, seed=0)
    pl_mask, _ = pick_nodes_random(data.train_mask, 0.1)
    logits = torch.rand(2708, 7)
    prompt_cora(0.5, pl_mask, logits, 2)
'''
# %%
data, num_classes = load_data(
            'cora', use_dgl=False, use_text=False, seed=0)

pl_mask, _ = pick_nodes_random(data.train_mask, 0.1)
logits = torch.rand(2708, 7)

prompt_cora_parallel(0.5, pl_mask, logits, 4)
ray.shutdown()
pl = answer_parser_cora(pl_mask)

prompt_cora(temperature, pl_mask, logits, gpu_nums)

# %%
ray.shutdown()
# %%
# set RAY_DEDUP_LOGS=0 to disable log deduplication
pl_mask_test_list = [189, 602, 633, 1099, 1379, 1380, 1650, 1689, 1775, 2295, 2659]
pl_mask_for_try = np.zeros(2708, dtype=bool)
pl_mask_for_try[pl_mask_test_list] = True
pl_mask_for_try = torch.tensor(pl_mask_for_try)
prompt_cora_parallel(0.5, pl_mask_for_try, logits, 1)
ray.shutdown()

# %%
print(pl_mask.size())
'''