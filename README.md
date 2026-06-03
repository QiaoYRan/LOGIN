# LOGIN

**LLM cOnsulted GNN traINing** — WSDM 2025 ([paper](https://doi.org/10.1145/3701551.3703488)).

Large Language Models are consulted on uncertain nodes during GNN training; responses drive semantic feature updates (correct LLM labels) and structure refinement (incorrect labels).

## Repository layout

```
LOGIN/
  core/                      # Python package
    train_lcgnn_with_ucscore.py   # main entry
    pipelines/lcgnn_ucscore_pipeline.py
    GNNs/                    # trainers + GCN/SAGE/MixHop/...
    LLMs/                    # prompt + explanation encoding
    data_utils/              # dataset loaders
    paths.py                 # LOGIN_TAPE_ROOT, etc.
  data/                      # symlink your TAPE assets here (gitignored)
  requirements.txt
```

## Setup

```bash
cd LOGIN
python3 -m pip install --user virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Link data (see [data/README.md](data/README.md)):

```bash
ln -s /your/path/to/TAPE data/TAPE
# or: export LOGIN_TAPE_ROOT=/your/path/to/TAPE
```

## Train

Run from the **LOGIN repo root** (parent of `core/`):

```bash
source .venv/bin/activate
export LOGIN_TAPE_ROOT="$(pwd)/data/TAPE"   # if using symlink

python3 -m core.train_lcgnn_with_ucscore \
  dataset cora \
  gnn.model.name GCN \
  gnn.train.feature_type TA \
  gnn.train.dropout 0.6 \
  gnn.train.pl_rate 0.1 \
  runs 1 \
  seed 0
```

Multi-run:

```bash
python3 -m core.train_lcgnn_with_ucscore \
  dataset cora gnn.model.name GCN gnn.train.feature_type TA \
  runs 5 seed null
```

## Pipeline (paper-aligned)

1. MC-dropout ensemble pretrain (`T=5`, same dropout rate).
2. Uncertainty on **train** nodes → top-`γ` consult LLM.
3. Feedback: correct → explanation embeddings; wrong → prune edges (vs ground truth).
4. Retrain GNN on augmented graph (temporary edge changes, restored after LC stage).
5. Node-wise ensemble of pretrain vs LC logits on pseudo-label nodes.

## What is **not** in this repo

- Raw graphs, prompt JSON, LM `.emb` files, LLM response cache (too large).
- Legacy experiment scripts from the old monolithic `core/` tree.
- RevGAT / DGL path (removed; use GCN, SAGE, MixHop, etc.).

## Citation

```bibtex
@inproceedings{qiao2025login,
  title={LOGIN: A Large Language Model Consulted Graph Neural Network Training Framework},
  author={Qiao, Yiran and Ao, Xiang and Liu, Yang and Xu, Jiarong and Sun, Xiaoqian and He, Qing},
  booktitle={WSDM},
  year={2025}
}
```

## Upload to GitHub

```bash
cd /storage/qiaoyr/LOGIN
git init
git add .
git commit -m "Initial public release of LOGIN training code"
git remote add origin git@github.com:YOUR_USER/LOGIN.git
git push -u origin main
```

Check size before push: `du -sh .` should be small (code only). Never `git add data/TAPE`.
