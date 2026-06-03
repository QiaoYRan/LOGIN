"""Data and asset paths for LOGIN. Override with LOGIN_DATA_ROOT."""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    return Path(os.environ.get("LOGIN_DATA_ROOT", _REPO_ROOT / "data"))


def dataset_root() -> Path:
    return data_root() / "dataset"


def prompts_root() -> Path:
    return data_root() / "prompts"


def prompt_path(dataset: str, node_idx: int) -> Path:
    return prompts_root() / dataset / f"{dataset}_{node_idx}.json"


def llm_output_dir(llm_name: str, dataset: str) -> Path:
    return prompts_root() / "LLMs" / llm_name / dataset


def llm_response_path(llm_name: str, dataset: str, node_idx: int) -> Path:
    return llm_output_dir(llm_name, dataset) / f"{dataset}_{node_idx}.json"


def prt_lm_path(dataset: str, lm_name: str, seed: int, variant: str = "TA") -> Path:
    sub = "prt_lm_1" if variant == "re" else "prt_lm"
    return data_root() / sub / dataset / f"{lm_name}-seed{seed}.emb"


def gpt_preds_path(dataset: str) -> Path:
    return data_root() / "gpt_preds" / f"{dataset}.csv"
