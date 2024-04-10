# %%
import sys
import os
sys.path.append(os.path.abspath('/storage/qiaoyr/TAPE'))
from core.GNNs.lcgnn_trainer import LCGNNTrainer
# from core.LLMs.prompt_llm import prompt_LLM, answer_parser
from core.LLMs.encode_explanations import encode_explanations_batch
from core.LLMs.prompt import prompt_LLM, answer_parser
import pandas as pd
from core.config import cfg, update_cfg
from core.data_utils.load import load_data
import time
import torch
import numpy as np
from core.utils import calculate_uncertainty_score, find_topk_uncertain_nodes, find_misclassified_nodes, ensemble_logits,  evaluate_with_logits
torch.cuda.init()
from core.LMs.lm_trainer import LMTrainer

def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)

   
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
        # GNN pretraining stage
        # train 5 GNNs with different dropout rates
        drop_out_list = [0.5, 0.5, 0.5, 0.5, 0.5]
        logits_list = []
        test_acc_list = []
        acc_list = []
        for dropout in drop_out_list:
            cfg.gnn.train.dropout = dropout
            trainer = TRAINER(cfg, cfg.gnn.train.feature_type, data, num_classes)
            trainer.train(prt_sign=True)
            logits, acc = trainer.eval_and_save()
            print(f'dropout:{dropout}, acc:{acc}')
            logits_list.append(logits)
            test_acc_list.append(acc['test_acc'])
            acc_list.append(acc)
        best_index = test_acc_list.index(max(test_acc_list))
        best_dropout = drop_out_list[best_index]
        pre_logits = logits_list[best_index]
        all_acc_prt.append(acc_list[best_index])
        probabilities = torch.softmax(pre_logits, dim=1)
        y_pred = torch.argmax(probabilities, dim=1)
        y_pred = y_pred.detach().cpu().numpy()
        uc_score = calculate_uncertainty_score(logits_list)
        pl_mask = find_topk_uncertain_nodes(uc_score, cfg.gnn.train.pl_rate)
        print("pl_nodes_num:", pl_mask.sum())
        pseudo_labels, explanations = prompt_LLM(cfg, pl_mask, pre_logits)
        pl_nodes_list = pl_mask.nonzero().squeeze().tolist()
        print(pl_nodes_list)
        print(len(text))
        for i, node in enumerate(pl_nodes_list):
            # explanations[i] = text[node] + explanations[i]
            if pseudo_labels[i] == data.y[node]: 
                text[node] = str(explanations[i] + text[node])


        encoder = LMTrainer(cfg, data, num_classes, text)
        encoder.eval_and_save()
        mid_path = f'/storage/qiaoyr/TAPE/mid_final/'
        file_name = f"{cfg.dataset}_seed{seed}_output.txt"
        with open(mid_path+file_name, 'w') as file:
            for item in text:
                file.write("%s\n" % item)

        #emb = encode_explanations_batch(explanations)
        pseudo_labels = np.array(pseudo_labels)
        #emb = emb.numpy()
        pl_mask = pl_mask.detach().cpu().numpy()
        save_path = os.path.join(mid_path, f'{cfg.dataset}_{cfg.gnn.model.name}_{seed}')
        np.savez(save_path, pseudo_labels = pseudo_labels, pl_mask = pl_mask)
        #print(pseudo_labels.shape, emb.shape, pl_mask.shape)

        # save pseudo labels, emb and pl_mask
        mid_path = f'/storage/qiaoyr/TAPE/mid_final/'
        save_path = os.path.join(mid_path, f'{cfg.dataset}_GCN_{seed}.npz')
        loaded_data = np.load(save_path)
        pseudo_labels = loaded_data['pseudo_labels']
        # emb = loaded_data['emb']
        pl_mask = loaded_data['pl_mask']
        pseudo_labels = pseudo_labels.tolist()
        # emb = torch.from_numpy(emb).to(torch.float32)
        pl_mask = torch.from_numpy(pl_mask).to(torch.bool)
        # update features
        trainer_lc = TRAINER(cfg, 're', data, num_classes)
        trainer_lc.set_pl_mask(pl_mask)
        #trainer_lc.update_pseudo_labels_and_features(pseudo_labels, emb)
        trainer_lc.update_pseudo_labels(pseudo_labels)
        # prune edges 
        trainer_lc.augment_adjacency_matrix_sim()
        trainer_lc.train(prt_sign=True)
        logits, acc = trainer_lc.eval_and_save()
        print(f'lc_acc:{acc}')
        all_acc_lc.append(acc)
       
    end = time.time()
    if len(all_acc_prt) > 1:
        df = pd.DataFrame(all_acc_prt)
        print(f"[{cfg.gnn.model.name} + {cfg.gnn.train.feature_type} + GNN] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")
    if len(all_acc_lc) > 1:
        df = pd.DataFrame(all_acc_lc)
        print(f"[{cfg.gnn.model.name} + {cfg.gnn.train.feature_type} + (GNN+LLM)] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")
    if len(all_acc_ensemble) > 1:
        df = pd.DataFrame(all_acc_ensemble)
        print(f"[{cfg.gnn.model.name} + {cfg.gnn.train.feature_type} + (GNN+LLM+ensemble)] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")
    print(f"Running time: {(end-start)/len(seeds):.2f}s")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)

# %%
