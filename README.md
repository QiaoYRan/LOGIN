# LOGIN

Implementation of **LOGIN: Large Language Model-Graph Oriented Interaction for Node Classification** (WSDM 2025).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place graph data, node prompts, LM embeddings, and LLM outputs under `data/` (see `data/README.md`), or point to an existing tree:

```bash
export LOGIN_DATA_ROOT=/path/to/your/assets
```

## Training

Run from the repository root:

```bash
python3 -m core.train_lcgnn_with_ucscore dataset cora gnn.model.name GCN gnn.train.feature_type TA seed 0
```

Common options (YACS `opts`):

| Option | Default | Description |
|--------|---------|-------------|
| `dataset` | `cora` | Dataset name |
| `gnn.model.name` | `GCN` | GNN backbone |
| `gnn.train.feature_type` | `TA` | Node features (`ogb`, `TA`, `llm`, …) |
| `gnn.train.pl_rate` | `0.1` | Fraction of train nodes for LLM pseudo-labels |
| `gnn.train.dropout` | `0.6` | Dropout (MC ensemble uses fixed ratio) |
| `llm.name` | `vicuna` | LLM for consultation |
| `seed` | random | Random seed |

## Layout

```
LOGIN/
├── core/                 # training code
│   ├── train_lcgnn_with_ucscore.py
│   ├── pipelines/
│   ├── GNNs/
│   ├── LLMs/
│   └── data_utils/
├── data/                 # assets (not shipped; see data/README.md)
└── requirements.txt
```

## Citation

```bibtex
@inproceedings{qiao2025login,
  title={Login: A large language model consulted graph neural network training framework},
  author={Qiao, Yiran and Ao, Xiang and Liu, Yang and Xu, Jiarong and Sun, Xiaoqian and He, Qing},
  booktitle={Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining},
  pages={232--241},
  year={2025}
}
```