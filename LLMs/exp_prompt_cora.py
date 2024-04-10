# %%
import torch
import torch.nn.functional as F
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import re
from conversation import conv_templates
import ray
import sys
import os


LLM_model = 'lmsys/vicuna-7b-v1.5'


category = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']

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

@torch.inference_mode()
def prompt_cora(cfg, pl_mask, logits):
    '''
    input: pl_mask, cfg(device, llm_name), gnn_predicted_labels(logits)
    '''
    probabilities = F.softmax(logits, dim=1)
    y_pred = torch.argmax(probabilities, dim=1)
    y_pred = y_pred.detach().cpu().numpy()
    nodes_num = np.size(y_pred)

    disable_torch_init()
    print('start loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(LLM_model)
    print('finish loading')

    print('start loading LM')
    model = AutoModelForCausalLM.from_pretrained(LLM_model, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    res_data = []
    for idx in tqdm(range(nodes_num)):
        prompt = prompt_cora_per_node(idx, y_pred)
        qs = prompt['conversations']['value']
        pattern = r'<graph>' 
        qs = re.sub(pattern, str(prompt['graph']), qs)
        conv_mode = "vicuna_v1_1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()
        print('prompt:', final_prompt)
        input_ids = tokenizer([final_prompt]).input_ids
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        print('answer:', outputs)
        res_data = {"id": prompt["id"], "node_idx": prompt["graph"]["node_idx"], "res": outputs}.copy()
        res_file = f'/storage/qiaoyr/TAPE/prompts/LLMs/vicuna/cora/cora_{idx}.json'
        with open(res_file, 'w') as f:
            json.dump(res_data, f)
# %%
import sys
import os
sys.path.append(os.path.abspath('/storage/qiaoyr/TAPE'))

from core.GNNs.gnn_utils import pick_nodes_random
from core.data_utils.load import load_data
data, num_classes = load_data(
            'cora', use_dgl=False, use_text=False, seed=0)

pl_mask, _ = pick_nodes_random(data.train_mask, 0.1)
logits = torch.rand(2708, 7)
prompt_cora(None, pl_mask, logits)



# %%

# %%
# %%
def prompt_cora_parallel(cfg, pl_mask, logits, num_gpus):
    probabilities = F.softmax(logits, dim=1)
    y_pred = torch.argmax(probabilities, dim=1)
    y_pred = y_pred.detach().cpu().numpy()
    nodes_num = np.size(y_pred)
    chunk_size = nodes_num // num_gpus
    ans_handles = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = nodes_num if i == num_gpus - 1 else (i + 1) * chunk_size
        ans_handles.append(prompt_cora_batch.remote(cfg, pl_mask, logits, start, end))

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.append(ray.get(ans_handle))
    return ans_jsons

# %%
@ray.remote(num_gpus=1)
@torch.inference_mode()
def prompt_cora_batch(cfg, pl_mask, logits, start, end):
    '''
    input: pl_mask, cfg(device, llm_name), gnn_predicted_labels(logits)
    '''
    probabilities = F.softmax(logits, dim=1)
    y_pred = torch.argmax(probabilities, dim=1)
    y_pred = y_pred.detach().cpu().numpy()
    nodes_num = np.size(y_pred)

    disable_torch_init()
    print('start loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(LLM_model)
    print('finish loading')

    print('start loading LM')
    model = AutoModelForCausalLM.from_pretrained(LLM_model, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    for idx in tqdm(range(start, end)):
        prompt = prompt_cora_per_node(idx, y_pred)
        qs = prompt['conversations']['value']
        pattern = r'<graph>' 
        qs = re.sub(pattern, str(prompt['graph']), qs)
        conv_mode = "vicuna_v1_1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()
        print('prompt:', final_prompt)
        input_ids = tokenizer([final_prompt]).input_ids
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        print('answer:', outputs)
        res_data = {"id": prompt["id"], "node_idx": prompt["graph"]["node_idx"], "res": outputs}.copy()
        res_file = f'/storage/qiaoyr/TAPE/prompts/LLMs/vicuna/parallel_test_cora_double/cora_{idx}.json'
        with open(res_file, 'w') as f:
            json.dump(res_data, f)
    return start
# %%
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
sys.path.append(os.path.abspath('/storage/qiaoyr/TAPE'))
ray.init()
from core.GNNs.gnn_utils import pick_nodes_random
from core.data_utils.load import load_data
data, num_classes = load_data(
            'cora', use_dgl=False, use_text=False, seed=0)

pl_mask, _ = pick_nodes_random(data.train_mask, 0.1)
logits = torch.rand(2708, 7)

prompt_cora_parallel(None, pl_mask, logits, 4)
ray.shutdown()


# %%
ray.shutdown()
# %%
