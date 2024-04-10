import torch
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('/storage/qiaoyr/TAPE'))
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from core.LMs.model import BertClassifier, BertClaInfModel
from core.data_utils.dataset import Dataset
from core.data_utils.load import load_data
from core.utils import init_path, time_logger


def encode_explanations_plm(cfg, explanations):
    model_name = "microsoft/deberta-base"
    print("loading encoder for explanations...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    model = BertClassifier(bert_model, n_labels=cfg.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prev_ckpt = f'prt_lm/{cfg.dataset}/{model_name}-seed{cfg.seed}.ckpt'
    print("Initialize using previous ckpt...")
    model.load_state_dict(torch.load(prev_ckpt))
    with torch.no_grad():

        inputs = tokenizer(explanations, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = model.bert_encoder(**inputs, output_hidden_states=True)

        # outputs[0]=last hidden state
        emb = model.dropout(outputs['hidden_states'][-1])
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
    return cls_token_emb