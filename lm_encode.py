from core.LMs.lm_trainer import LMTrainer
from core.config import cfg, update_cfg
import pandas as pd
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4'
sys.path.append(os.path.abspath('/storage/qiaoyr/TAPE'))
from core.GNNs.gnn_trainer import GNNTrainer
from core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
from core.GNNs.llm_gnn_trainer import LLMGNNTrainer
from core.GNNs.lcgnn_trainer import LCGNNTrainer
# from core.LLMs.prompt_llm import prompt_LLM, answer_parser
from core.LLMs.encode_explanations import encode_explanations
from core.LLMs.prompt import prompt_LLM, answer_parser
import pandas as pd
from core.config import cfg, update_cfg
from core.data_utils.load import load_data
import time
import torch
import numpy as np
from core.utils import calculate_uncertainty_score, find_topk_uncertain_nodes, find_misclassified_nodes, ensemble_logits,  evaluate_with_logits
torch.cuda.init()

def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_acc = []
    for seed in seeds:
        cfg.seed = seed
        data, num_classes, text = load_data(cfg.dataset, use_dgl=False, use_text=True, seed=seed)
        trainer = LMTrainer(cfg, data, num_classes, text)
        # trainer.train()
        acc = trainer.eval_and_save()
        all_acc.append(acc)

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        for k, v in df.items():
            print(f"{k}: {v.mean():.4f} ± {v.std():.4f}")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)
