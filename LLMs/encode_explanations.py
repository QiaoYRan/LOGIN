

# %%
import os
import torch
def get_pl_mask():
    folder_path = '/storage/qiaoyr/TAPE/prompts/LLMs/vicuna/cora_test' 
    pl_nodes_list = []
    for f in os.listdir(folder_path):
        pl_nodes_list.append(int(f.split('_')[-1].split('.')[0]))
    pl_nodes_list = sorted(pl_nodes_list)
    pl_mask = torch.zeros(2708, dtype=torch.bool)
    pl_mask[pl_nodes_list] = True
    return pl_mask

# %%
import sys
import os
sys.path.append(os.path.abspath('/storage/qiaoyr/TAPE'))
from transformers import AutoTokenizer, AutoModel
from core.LLMs.prompt_llm import prompt_LLM, answer_parser
from core.LLMs.prompt_cora import answer_parser_cora
'''
def encode_explanations(explanations):
    # explanations is a list of strings
    # pl_mask = get_pl_mask() 
    # pl, explanations = answer_parser_cora(pl_mask)
    model_name = "microsoft/deberta-base"
    print('loading encoder for explanations...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    with torch.no_grad():
        inputs = tokenizer(explanations, padding=True, truncation=True, max_length=512, return_tensors="pt")
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
    
    return cls_emb
'''
import torch

def encode_explanations(explanations):
    # explanations is a list of strings
    # pl_mask = get_pl_mask() 
    # pl, explanations = answer_parser_cora(pl_mask)
    model_name = "microsoft/deberta-base"
    print('loading encoder for explanations...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        # batched inputs

        inputs = tokenizer(explanations, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].to("cpu")  # 将嵌入移到CPU上
    
    return cls_emb

import torch
from transformers import AutoTokenizer, AutoModel

def encode_explanations_batch(explanations, batch_size=128):
    # explanations is a list of strings
    model_name = "microsoft/deberta-base"
    print('loading encoder for explanations...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cls_embs = []  # 用于存储每个批次的CLS嵌入向量

    with torch.no_grad():
        for i in range(0, len(explanations), batch_size):
            batch_explanations = explanations[i:i+batch_size]
            inputs = tokenizer(batch_explanations, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].to("cpu")
            cls_embs.append(cls_emb)

    # 将所有批次的嵌入向量连接在一起
    cls_embeddings = torch.cat(cls_embs, dim=0)

    return cls_embeddings




def encode_explanations_plm(explanations):
    # explanations is a list of strings
    # pl_mask = get_pl_mask() 
    # pl, explanations = answer_parser_cora(pl_mask)
    model_name = "microsoft/deberta-base"
    print('loading encoder for explanations...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        inputs = tokenizer(explanations, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].to("cpu")  # 将嵌入移到CPU上
    torch.cuda.empty_cache()
    return cls_emb

# %%
# print(cls_emb.shape)
# print(outputs.last_hidden_state.shape)
# print(len(explanations))
'''
torch.Size([162, 768])
torch.Size([162, 246, 768])
162
'''
# %%
#inputs = tokenizer(explanations[0], padding=True, truncation=True, max_length=512)
#tokenizer.decode(inputs["input_ids"])

