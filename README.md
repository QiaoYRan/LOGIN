# LOGIN

**LLM cOnsulted GNN traINing** — WSDM 2025 ([paper](https://doi.org/10.1145/3701551.3703488)).

Large Language Models are consulted on uncertain nodes during GNN training; responses drive semantic feature updates (correct LLM labels) and structure refinement (incorrect labels).

## Repository layout

This repo contains:

- **Recommended (new):** `core/` package with the maintained pipeline
- **Legacy (root):** older scripts such as `login.py`, top-level `GNNs/`, `LLMs/` (kept for backward compatibility)

```
LOGIN/
  core/                      # recommended entry
    train_lcgnn_with_ucscore.py
    pipelines/lcgnn_ucscore_pipeline.py
    GNNs/ LLMs/ data_utils/
  login.py                   # legacy entry
  GNNs/ LLMs/ LMs/           # legacy modules
  data/                      # symlink TAPE assets here (gitignored)
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

## Train (recommended)

Run from the **repo root**:

```bash
source .venv/bin/activate
export LOGIN_TAPE_ROOT="$(pwd)/data/TAPE"

python3 -m core.train_lcgnn_with_ucscore \
  dataset cora \
  gnn.model.name GCN \
  gnn.train.feature_type TA \
  gnn.train.dropout 0.6 \
  gnn.train.pl_rate 0.1 \
  runs 1 \
  seed 0
```

## Pipeline (paper-aligned)

1. MC-dropout ensemble pretrain (`T=5`, same dropout rate).
2. Uncertainty on **train** nodes → top-`γ` consult LLM.
3. Feedback: correct → explanation embeddings; wrong → prune edges (vs ground truth).
4. Retrain GNN on augmented graph (temporary edge changes, restored after LC stage).
5. Node-wise ensemble of pretrain vs LC logits on pseudo-label nodes.

## Citation

```bibtex
@inproceedings{qiao2025login,
  title={LOGIN: A Large Language Model Consulted Graph Neural Network Training Framework},
  author={Qiao, Yiran and Ao, Xiang and Liu, Yang and Xu, Jiarong and Sun, Xiaoqian and He, Qing},
  booktitle={WSDM},
  year={2025}
}
```
