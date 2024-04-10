from core.GNNs.gnn_trainer import GNNTrainer
from core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
from core.GNNs.llm_gnn_trainer import LLMGNNTrainer
from core.GNNs.base_trainer import LCGNNTrainer
import pandas as pd
from core.config import cfg, update_cfg
import time
from core.data_utils.load import load_data

def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    print(cfg)
    if cfg.gnn.model.name == 'RevGAT':
        TRAINER = DGLGNNTrainer
    else:
        TRAINER = LCGNNTrainer

    all_acc = []
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        data, num_classes, text = load_data(cfg.dataset, use_dgl=False, use_text=True, seed=seed)
        trainer = TRAINER(cfg, cfg.gnn.train.feature_type, data, num_classes)
        trainer.train(prt_sign=True)
        logits, acc = trainer.eval_and_save()
        all_acc.append(acc)
    end = time.time()

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        print(f"[{cfg.gnn.model.name} + {cfg.gnn.train.feature_type}] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")
    print(f"Running time: {(end-start)/len(seeds):.2f}s")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)
