# %%
import sys
import os
sys.path.append(os.path.abspath('/storage/qiaoyr/TAPE'))
from core.GNNs.gnn_trainer import GNNTrainer
from core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
from core.GNNs.llm_gnn_trainer import LLMGNNTrainer
from core.GNNs.lcgnn_trainer import LCGNNTrainer
# from core.LLMs.prompt_llm import prompt_LLM, answer_parser
from core.LLMs.encode_explanations import encode_explanations_batch
from core.LLMs.prompt_no_gnn import prompt_LLM, answer_parser
import pandas as pd
from core.config import cfg, update_cfg
from core.data_utils.load import load_data
import time
import torch
import numpy as np
from core.utils import calculate_uncertainty_score, find_topk_uncertain_nodes, find_misclassified_nodes, ensemble_logits,  evaluate_with_logits
torch.cuda.init()
from core.LMs.lm_trainer import LMTrainer
import numpy as np
import torch
def pick_nodes_random(train_mask, num, seed=0):

    np.random.seed(seed)
    true_indices = torch.nonzero(train_mask).squeeze() 
    num_samples = num
    if num_samples > 0:
        sampled_indices = np.random.choice(true_indices.cpu().numpy(), size=num_samples, replace=False)
        extracted_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        extracted_mask[sampled_indices] = True
   

    return extracted_mask

def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)

    if cfg.gnn.model.name == 'RevGAT':
        TRAINER = DGLGNNTrainer #TODO
    else:
        TRAINER = LCGNNTrainer
 
    all_acc_prt = []
    all_acc_lc = []
    all_acc_ensemble = []
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        print('cfg:', cfg)
        # get dataset split at the very beginning
        data, num_classes, text = load_data(cfg.dataset, use_dgl=False, use_text=True, seed=seed)
        print(data.y)
        data.test_mask = pick_nodes_random(data.test_mask, 500, seed=seed)
        pseudo_labels, explanations = prompt_LLM(cfg, data.test_mask)
        pseudo_labels = np.array(pseudo_labels)
        ground_truth = data.y[data.test_mask].numpy()
        # cal acc
        acc = (pseudo_labels == ground_truth).sum() / len(ground_truth)
        print(f'{cfg.dataset} llm prediction acc: {acc}')
        all_acc_prt.append({'acc': acc})
    if len(all_acc_prt) > 1:
        df = pd.DataFrame(all_acc_prt)
        print(f"[{cfg.gnn.model.name} + {cfg.gnn.train.feature_type} + GNN] ACC: {df['acc'].mean():.4f} ± {df['acc'].std():.4f}")
    end = time.time()
    if len(all_acc_prt) > 1:
        df = pd.DataFrame(all_acc_prt)
    print(f"Running time: {(end-start)/len(seeds):.2f}s")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)

# %%
