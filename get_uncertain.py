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
        # GNN pretraining stage
        # train 5 GNNs with different dropout rates
        #  cora: 0,0.1,0.4,0.6,0.8
        # drop_out_list = [0, 0.1, 0.4, 0.6, 0.8]
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
        # find the best dropout rate
        best_index = test_acc_list.index(max(test_acc_list))
        best_dropout = drop_out_list[best_index]
        print(f'best_dropout:{best_dropout}, acc:{max(test_acc_list)}')
        pre_logits = logits_list[best_index]
        all_acc_prt.append(acc_list[best_index])
        # get pretrained GNN prediction
        probabilities = torch.softmax(pre_logits, dim=1)
        y_pred = torch.argmax(probabilities, dim=1)
        y_pred = y_pred.detach().cpu().numpy()
        # select the most uncertain nodes
        uc_score = calculate_uncertainty_score(logits_list)
        pl_mask = find_topk_uncertain_nodes(uc_score, cfg.gnn.train.pl_rate)
        print("pl_nodes_num:", pl_mask.sum())
        # prompt LLM for pseudo labels of the most uncertain nodes  
        pseudo_labels, explanations = prompt_LLM(cfg, pl_mask, pre_logits)
        # pseudo_labels, explanations = answer_parser(cfg, pl_mask)
        # encode explanations
        pl_nodes_list = pl_mask.nonzero().squeeze().tolist()
        print(pl_nodes_list)
        print(len(text))
        for i, node in enumerate(pl_nodes_list):
            # explanations[i] = text[node] + explanations[i]
            if pseudo_labels[i] == data.y[node]: 
                text[node] = str(explanations[i] + text[node])
                #print(text[node])
        ####################

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
        '''
        # save pseudo labels, emb and pl_mask
        mid_path = f'/storage/qiaoyr/TAPE/mid_final/'
        save_path = os.path.join(mid_path, f'{cfg.dataset}_{cfg.gnn.model.name}_{seed}')
        np.savez(save_path, pseudo_labels = pseudo_labels, emb = emb, pl_mask = pl_mask)
        # save pseudo labels and emb and pl_mask 
        pseudo_labels = np.array(pseudo_labels)
        emb = emb.numpy()
        pl_mask = pl_mask.detach().cpu().numpy()
        print(pseudo_labels.shape, emb.shape, pl_mask.shape)
        # save pseudo labels, emb and pl_mask
        mid_path = f'/storage/qiaoyr/TAPE/mid_final/'
        save_path = os.path.join(mid_path, f'{cfg.dataset}_{cfg.gnn.model.name}_{seed}')
        np.savez(save_path, pseudo_labels = pseudo_labels, emb = emb, pl_mask = pl_mask)
    
        # update pseudo labels and features encoded from llm explanations
        cfg.gnn.train.epochs += 200
        cfg.gnn.train.dropout = best_dropout
        trainer_lc = TRAINER(cfg, cfg.gnn.train.feature_type, data, num_classes)
        trainer_lc.set_pl_mask(pl_mask)
        trainer_lc.update_pseudo_labels_and_features(pseudo_labels, emb)
        # prune edges 
        trainer_lc.augment_adjacency_matrix_sim()
        trainer_lc.train(prt_sign=True)
        logits, acc = trainer_lc.eval_and_save()
        print(f'lc_acc:{acc}')
        all_acc_lc.append(acc)
        new_logits = ensemble_logits(pre_logits, logits, pl_mask.to(trainer.device))
        misclassified_nodes = find_misclassified_nodes(new_logits, data.y)
        
        acc_ensemble = evaluate_with_logits(new_logits, trainer)
        print(f'final_acc:{acc_ensemble}')
        all_acc_ensemble.append(acc_ensemble)
        '''

       
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
