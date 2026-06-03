# External data (not included in git)

LOGIN needs a TAPE-style data directory. **Do not commit** prompts, datasets, or embedding files to GitHub.

## Quick setup on your machine

If you already have TAPE under `/storage/qiaoyr/TAPE`:

```bash
cd /path/to/LOGIN
ln -s /storage/qiaoyr/TAPE data/TAPE
```

Or set an environment variable (no symlink):

```bash
export LOGIN_TAPE_ROOT=/storage/qiaoyr/TAPE
```

## Expected layout under `TAPE_ROOT`

```
TAPE/
  prompts/<dataset>/<dataset>_<node_id>.json
  prompts/LLMs/<llm_name>/<dataset>/<dataset>_<node_id>.json   # LLM outputs (cache)
  prt_lm/<dataset>/<lm_name>-seed<seed>.emb                  # LM node embeddings (TA features)
  dataset/                                                   # Cora, PubMed, WebKB, OGB, etc.
```

## Optional: pre-generate LLM cache

`prompt.py` only calls the LLM for nodes in `pl_mask`. You can pre-fill
`prompts/LLMs/vicuna/<dataset>/` for all nodes once, then training will read cache without re-querying.
