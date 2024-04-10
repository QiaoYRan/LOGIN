# %%
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
sys.path.append(os.path.abspath('/storage/qiaoyr/TAPE'))
from core.GNNs.gnn_trainer import GNNTrainer
from core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
from core.GNNs.llm_gnn_trainer import LLMGNNTrainer
# from core.LLMs.prompt_llm import prompt_LLM, answer_parser
from core.LLMs.encode_explanations import encode_explanations
from core.LLMs.prompt import prompt_LLM, answer_parser
import pandas as pd
from core.config import cfg, update_cfg
import time


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)

    if cfg.gnn.model.name == 'RevGAT':
        TRAINER = DGLGNNTrainer #TODO
    else:
        TRAINER = LLMGNNTrainer
 
    all_acc_prt = []
    all_acc_pl = []
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        trainer = TRAINER(cfg, cfg.gnn.train.feature_type)
        # GNN pretraining stage
        # train()函数中体现是否用pl，trainer始终为同一个
        trainer.train(prt_sign=True)
        logits, acc = trainer.eval_and_save()
        all_acc_prt.append(acc)
        print(acc)
        # prompt LLM for pseudo labels
        pseudo_labels, explanations = prompt_LLM(cfg, trainer.pl_mask, logits)  # TODO
        # pseudo_labels, explanations = answer_parser(cfg, trainer.pl_mask)  #for debugging
        # encode explanations
        emb = encode_explanations(explanations)
        # update pseudo labels and features encoded from llm explanations
        trainer = TRAINER(cfg, cfg.gnn.train.feature_type)
        trainer.update_pseudo_labels_and_features(pseudo_labels, emb)
        # train GNN with pl
        trainer.train(prt_sign=False) # TODO
        _, acc = trainer.eval_and_save()
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
