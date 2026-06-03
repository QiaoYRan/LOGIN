# Data layout

Large files are not included in the repository. Organize assets as follows (or set `LOGIN_DATA_ROOT` to a directory with the same structure):

```
data/
├── dataset/              # raw graphs and text
│   ├── cora_orig/
│   ├── PubMed_orig/
│   ├── web_kb/
│   ├── ogbn_arxiv/
│   └── ...
├── prompts/              # per-node prompt JSON
│   └── <dataset>/
├── prompts/LLMs/         # cached LLM responses
│   └── <llm_name>/<dataset>/
├── prt_lm/               # pretrained LM node embeddings (*.emb)
│   └── <dataset>/
└── gpt_preds/            # optional top-k predictions (*.csv)
```

Precomputed embeddings and prompts from your experiments can be symlinked into `data/` if they already live elsewhere on disk.
