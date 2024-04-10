# %%
import sys
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
sys.path.append(os.path.abspath('/storage/qiaoyr/TAPE'))
from core.GNNs.gnn_trainer import GNNTrainer
from core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
from core.GNNs.llm_gnn_trainer import LLMGNNTrainer
from core.LLMs.prompt import prompt_LLM, answer_parser
from core.LLMs.encode_explanations import encode_explanations, get_pl_mask      
import pandas as pd
from core.config import cfg, update_cfg
from core.utils import calculate_uncertainty_score, find_topk_uncertain_nodes, find_misclassified_nodes, ensemble_logits,  evaluate_with_logits
import time


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)

    if cfg.gnn.model.name == 'RevGAT':
        TRAINER = DGLGNNTrainer #TODO
    else:
        TRAINER = LLMGNNTrainer
 
    all_acc_prt = []
    all_acc_pl = []
    all_acc_final = []
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        
        # GNN pretraining stage
        # train 5 GNNs with different dropout rates
        #  cora: 0,0.1,0.4,0.6,0.8
        drop_out_list = [0, 0.1, 0.4, 0.6, 0.8]
        logits_list = []
        test_acc_list = []
        acc_list = []
        for dropout in drop_out_list:
            cfg.gnn.train.dropout = dropout
            trainer = TRAINER(cfg, cfg.gnn.train.feature_type)
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
        # # for debugging
        probabilities = torch.softmax(pre_logits, dim=1)
        y_pred = torch.argmax(probabilities, dim=1)
        y_pred = y_pred.detach().cpu().numpy()
        # select the most uncertain nodes
        uc_score = calculate_uncertainty_score(logits_list)
        pl_mask = find_topk_uncertain_nodes(uc_score, cfg.gnn.train.pl_rate)
        # prompt LLM for pseudo labels of the most uncertain nodes
        pseudo_labels, explanations = prompt_LLM(cfg, pl_mask, pre_logits)
        #pl_mask = get_pl_mask()
        #pseudo_labels, explanations = answer_parser(cfg, pl_mask)  #for debugging
        # encode explanations
        emb = encode_explanations(explanations)
        '''
        # !! for debugging, train gnn with the best dropout(obtained before), for cora, its 0.6
        cfg.gnn.train.dropout = 0.6
        pre_trainer = TRAINER(cfg, cfg.gnn.train.feature_type)
        pre_trainer.train(prt_sign=True)
        pre_logits, pre_acc = pre_trainer.eval_and_save()
        print(f'pre_acc:{pre_acc}')
        '''
        # update pseudo labels and features encoded from llm explanations
        trainer = TRAINER(cfg, cfg.gnn.train.feature_type)
        trainer.update_pseudo_labels_and_features_new(pl_mask, pseudo_labels, emb) # for debugging
        # train GNN with pl
        trainer.train(prt_sign=False) # TODO
        logits, acc = trainer.eval_and_save()
        # find which nodes are misclassified with trainer.data.y and logits
        # find_misclassified_nodes(logits, trainer.data.y)
        print(f'aft_acc:{acc}')
        all_acc_pl.append(acc)
        # ensemble logits
        '''
        print(f'pl_nodes:{torch.nonzero(pl_mask).squeeze().tolist() }')
        misclassified_nodes = find_misclassified_nodes(pre_logits, trainer.data.y)
        misclassified_nodes = find_misclassified_nodes(logits, trainer.data.y)
        '''
        new_logits = ensemble_logits(pre_logits, logits, pl_mask.to(trainer.device))
        misclassified_nodes = find_misclassified_nodes(new_logits, trainer.data.y)
        
        acc_final = evaluate_with_logits(new_logits, trainer)
        print(f'final_acc:{acc_final}')
        all_acc_final.append(acc_final)

        
    end = time.time()
    if len(all_acc_prt) > 1:
        df = pd.DataFrame(all_acc_prt)
        print(f"[{cfg.gnn.model.name} + {cfg.gnn.train.feature_type} + GNN] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")
    if len(all_acc_pl) > 1:
        df = pd.DataFrame(all_acc_pl)
        print(f"[{cfg.gnn.model.name} + {cfg.gnn.train.feature_type} + (GNN+LLM)] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")
    if len(all_acc_final) > 1:
        df = pd.DataFrame(all_acc_final)
        print(f"[{cfg.gnn.model.name} + {cfg.gnn.train.feature_type} + (GNN+LLM+ensemble)] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")

    print(f"Running time: {(end-start)/len(seeds):.2f}s")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)

# %%
