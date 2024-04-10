 for data in 'wisconsin' 
 do 
    python -m core.train_lcgnn_with_ucscore dataset $data gnn.train.feature_type 'TA' gnn.model.name 'MixHop' gnn.train.pl_rate 0.8 >> ${data}_MixHop.out
 done

 for data in 'cora' 
 do 
    python -m core.train_lcgnn_with_ucscore dataset $data gnn.train.feature_type 'TA' gnn.model.name 'GCN'  >> ${data}_GCN.out
 done
