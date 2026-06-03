import time
from typing import Dict, List

import pandas as pd
import torch

from core.GNNs.lcgnn_trainer import LCGNNTrainer
from core.LLMs.encode_explanations import encode_explanations
from core.LLMs.prompt import prompt_LLM
from core.data_utils.load import load_data
from core.utils import (
    calculate_uncertainty_score,
    ensemble_logits,
    evaluate_with_logits,
)


def _print_multi_run_stats(tag: str, all_acc: List[Dict[str, float]]) -> None:
    if len(all_acc) <= 1:
        return
    df = pd.DataFrame(all_acc)
    print(
        f"[{tag}] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, "
        f"TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}"
    )


def run_lcgnn_with_ucscore(cfg) -> None:
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    trainer_cls = LCGNNTrainer
    base_epochs = cfg.gnn.train.epochs

    all_acc_prt = []
    all_acc_lc = []
    all_acc_ensemble = []

    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        cfg.gnn.train.epochs = base_epochs
        print("cfg:", cfg)

        data, num_classes, _ = load_data(cfg.dataset, use_dgl=False, use_text=True, seed=seed)

        # Stage 1: pretrain an ensemble with fixed dropout ratio.
        dropout_ratio = cfg.gnn.train.dropout
        ensemble_size = 5
        logits_list = []
        trainer_ref = None
        for member_idx in range(ensemble_size):
            cfg.gnn.train.dropout = dropout_ratio
            trainer = trainer_cls(cfg, cfg.gnn.train.feature_type, data, num_classes)
            trainer.train(prt_sign=True)
            logits, acc = trainer.eval_and_save()
            print(f"member:{member_idx}, dropout:{dropout_ratio}, acc:{acc}")
            logits_list.append(logits)
            trainer_ref = trainer

        pre_logits = torch.mean(torch.stack(logits_list), dim=0)
        pre_acc = evaluate_with_logits(pre_logits, trainer_ref)
        all_acc_prt.append(pre_acc)
        print(f"pretrain_ensemble_acc:{pre_acc}")

        # Stage 2: select uncertain nodes and query LLM pseudo labels.
        uc_score = calculate_uncertainty_score(logits_list)
        train_indices = torch.nonzero(data.train_mask, as_tuple=False).view(-1)
        k = int(cfg.gnn.train.pl_rate * train_indices.numel())
        pl_mask = torch.zeros_like(data.train_mask, dtype=torch.bool)
        if k > 0:
            _, topk_local_idx = torch.topk(uc_score[train_indices], k)
            pl_mask[train_indices[topk_local_idx]] = True
        print("pl_nodes_num:", pl_mask.sum())

        pseudo_labels, explanations = prompt_LLM(cfg, pl_mask, pre_logits)
        emb = encode_explanations(explanations)

        # Stage 3: retrain with pseudo labels and explanation features.
        cfg.gnn.train.epochs = base_epochs + 200
        cfg.gnn.train.dropout = dropout_ratio
        trainer_lc = trainer_cls(cfg, cfg.gnn.train.feature_type, data, num_classes)
        trainer_lc.set_pl_mask(pl_mask)
        trainer_lc.update_pseudo_labels_and_features(pseudo_labels, emb)
        trainer_lc.augment_adjacency_matrix_sim()
        trainer_lc.train(prt_sign=False)
        logits, acc = trainer_lc.eval_and_save()
        trainer_lc.restore_original_graph()
        all_acc_lc.append(acc)
        print(f"lc_acc:{acc}")

        # Stage 4: node-wise ensemble between pretrained and LC logits.
        new_logits = ensemble_logits(pre_logits, logits, pl_mask.to(trainer_lc.device))
        acc_ensemble = evaluate_with_logits(new_logits, trainer_lc)
        all_acc_ensemble.append(acc_ensemble)
        print(f"final_acc:{acc_ensemble}")

    end = time.time()

    model_tag = f"{cfg.gnn.model.name} + {cfg.gnn.train.feature_type}"
    _print_multi_run_stats(f"{model_tag} + GNN", all_acc_prt)
    _print_multi_run_stats(f"{model_tag} + (GNN+LLM)", all_acc_lc)
    _print_multi_run_stats(f"{model_tag} + (GNN+LLM+ensemble)", all_acc_ensemble)
    print(f"Running time: {(end-start)/len(seeds):.2f}s")
