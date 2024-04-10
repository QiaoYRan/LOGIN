# %%
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
sys.path.append(os.path.abspath('/storage/qiaoyr/TAPE'))
from core.GNNs.gnn_trainer import GNNTrainer
from core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
from core.GNNs.llm_gnn_trainer import LLMGNNTrainer
from core.GNNs.lcgnn_trainer import LCGNNTrainer
# from core.LLMs.prompt_llm import prompt_LLM, answer_parser
from core.LLMs.encode_explanations import encode_explanations
from core.LMs.lm_encoder import encode_explanations_plm
from core.LLMs.prompt import prompt_LLM, answer_parser
import pandas as pd
from core.config import cfg, update_cfg
from core.data_utils.load import load_data
import time
import torch
torch.cuda.init()
def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)

    if cfg.gnn.model.name == 'RevGAT':
        TRAINER = DGLGNNTrainer #TODO
    else:
        TRAINER = LCGNNTrainer
 
    all_acc_prt = []
    all_acc_pl = []
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        print('cfg:', cfg)
        # get dataset split at the very beginning
        data, num_classes, text = load_data(
            cfg.dataset, use_dgl=False, use_text=True, seed=seed)
        print(data.y)
        cfg.num_classes = num_classes
        # GNN pretraining stage
        trainer = TRAINER(cfg, cfg.gnn.train.feature_type, data, num_classes)
        trainer.train(prt_sign=True)
        logits, acc = trainer.eval_and_save()
        all_acc_prt.append(acc)
        print(acc)
        # prompt LLM for pseudo labels
        pseudo_labels, explanations = prompt_LLM(cfg, trainer.pl_mask, logits)  # TODO
        # pseudo_labels, explanations = answer_parser(cfg, trainer.pl_mask)  #for debugging
        # encode explanations
        emb = encode_explanations(explanations) #0.8801
        # emb = encode_explanations_plm(cfg, explanations) 0.8782
        # update pseudo labels and features encoded from llm explanations
        trainer_lc = TRAINER(cfg, cfg.gnn.train.feature_type, data, num_classes)
        trainer_lc.update_pseudo_labels_and_features(pseudo_labels, emb)
        # trainer_lc.augment_adjacency_matrix()
        # trainer_lc.augment_adjacency_matrix_sim()
        # train GNN with pl
        trainer_lc.train(prt_sign=True) # TODO
        _, acc = trainer_lc.eval_and_save()
        print(acc)
        all_acc_pl.append(acc)
    end = time.time()
    if len(all_acc_prt) > 1:
        df = pd.DataFrame(all_acc_prt)
        print(f"[{cfg.gnn.model.name} + {cfg.gnn.train.feature_type} + GNN] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")
    if len(all_acc_pl) > 1:
        df = pd.DataFrame(all_acc_pl)
        print(f"[{cfg.gnn.model.name} + {cfg.gnn.train.feature_type} + (GNN+LLM)] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")
    print(f"Running time: {(end-start)/len(seeds):.2f}s")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)

# %%
