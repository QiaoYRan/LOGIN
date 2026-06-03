"""Configurable data roots for LOGIN (set via environment variables)."""
import os

# Repository root: LOGIN/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# External TAPE assets (prompts, LM embeddings, datasets). Not shipped in git.
TAPE_ROOT = os.environ.get("LOGIN_TAPE_ROOT", os.path.join(PROJECT_ROOT, "data", "TAPE"))

DATASET_ROOT = os.environ.get("LOGIN_DATASET_ROOT", os.path.join(TAPE_ROOT, "dataset"))

PROMPTS_ROOT = os.path.join(TAPE_ROOT, "prompts")
LLM_RESPONSES_ROOT = os.path.join(PROMPTS_ROOT, "LLMs")
PRT_LM_ROOT = os.path.join(TAPE_ROOT, "prt_lm")


def llm_response_dir(llm_name: str, dataset: str) -> str:
    return os.path.join(LLM_RESPONSES_ROOT, llm_name, dataset)


def node_prompt_path(dataset: str, node_idx: int) -> str:
    return os.path.join(PROMPTS_ROOT, dataset, f"{dataset}_{node_idx}.json")


def prt_lm_emb_path(dataset: str, lm_name: str, seed: int) -> str:
    return os.path.join(PRT_LM_ROOT, dataset, f"{lm_name}-seed{seed}.emb")
