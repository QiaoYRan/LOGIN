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
LLM_model_dict = {'vicuna': 'lmsys/vicuna-7b-v1.5'}

@torch.inference_mode()

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def prompt_webkb_per_node(dataset, node_idx, y_pred):
    category = ['course', 'faculty', 'student','project', 'staff']
    file_path = f'/storage/qiaoyr/TAPE/prompts/{dataset}/{dataset}_{node_idx}.json'
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
def prompt_webkb_batch(dataset, llm_name, temperature, logits, pl_mask):
    '''
    input: temperature, gnn_predicted_labels(logits), nodes in this batch
    '''
    # get GNN predicted labels
    # probabilities = F.softmax(logits, dim=1)
    probabilities = torch.softmax(logits, dim=1)
    y_pred = torch.argmax(probabilities, dim=1)
    y_pred = y_pred.detach().cpu().numpy()

    nodes = torch.nonzero(pl_mask).squeeze().tolist()  
    LLM_model = LLM_model_dict[llm_name]
    disable_torch_init()
    print('start loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(LLM_model)
    print('finish loading')

    print('start loading LM')
    model = AutoModelForCausalLM.from_pretrained(LLM_model, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    for idx in tqdm(nodes):
        print(idx)  # TODO: remove
        prompt = prompt_webkb_per_node(dataset, idx, y_pred)
        qs = prompt['conversations']['value']
        pattern = r'<graph>' 
        qs = re.sub(pattern, str(prompt['graph']), qs)
        if llm_name == 'vicuna':
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
        folder_path = f'/storage/qiaoyr/TAPE/prompts/LLMs/{llm_name}/{dataset}'
        res_file = folder_path + f'/{dataset}_{idx}.json'
        with open(res_file, 'w') as f:
            json.dump(res_data, f)
    return idx


# %%

def answer_parser_webkb_per_node(dataset, llm_name, idx):
    category_list = ['course', 'faculty', 'student','project', 'staff']
    file_path = f'/storage/qiaoyr/TAPE/prompts/LLMs/{llm_name}/{dataset}/{dataset}_{idx}.json'
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
    try:
        explanation = dict_ans['explanation']
    except:
        print(f'cant resolve explanation for idx {idx}')
        explanation = ''
    return label, explanation

def answer_parser_webkb(dataset, llm_name, pl_mask):
    pl_nodes_list = torch.nonzero(pl_mask).squeeze().tolist()  
    print(pl_nodes_list)
    pseudo_labels = []
    explanations = []
    for i in pl_nodes_list:
        label, explanation = answer_parser_webkb_per_node(dataset, llm_name, i)
        pseudo_labels.append(label)
        explanations.append(explanation)
    # pseudo_labels = torch.tensor(pseudo_labels)
    return pseudo_labels, explanations

# %%
def prompt_webkb(dataset, llm_name, temperature, pl_mask, logits, gpu_nums):
    # data, num_classes = load_data('cora', use_dgl=False, use_text=False, seed=0)
    folder_path = f'/storage/qiaoyr/TAPE/prompts/LLMs/{llm_name}/{dataset}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with torch.no_grad():
        prompt_webkb_batch(dataset, llm_name, temperature, logits, pl_mask)
    pl, explanations = answer_parser_webkb(dataset, llm_name, pl_mask)
    print(pl)
    return pl, explanations

'''
# %%
dataset = 'wisconsin'
llm_name = 'vicuna'
data, num_classes = load_data(
            dataset, use_dgl=False, use_text=False, seed=0)
# %%
print(data)
# %%
pl_mask, _ = pick_nodes_random(data.train_mask, 0.1)
logits = torch.rand(data.x.shape[0], num_classes)
prompt_webkb(dataset, llm_name, 0.5, pl_mask, logits, 2)

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



