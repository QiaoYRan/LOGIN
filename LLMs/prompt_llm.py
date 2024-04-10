import sys
sys.path.append('/storage/qiaoyr/TAPE')
from core.LLMs.prompt_cora import prompt_cora, answer_parser_cora
from core.LLMs.prompt_webkb import prompt_webkb, answer_parser_webkb

def prompt_LLM(cfg, pl_mask, logits):
    if cfg.dataset == 'cora':
        pl, explanations = prompt_cora(cfg.llm.temperature, pl_mask, logits, cfg.llm.gpu_nums)
    if cfg.dataset == 'wisconsin' or 'cornell' or 'texas':
        pl, explanations = prompt_webkb(cfg.dataset, cfg.llm.name, cfg.llm.temperature, pl_mask, logits, cfg.llm.gpu_nums)
    return pl, explanations

def answer_parser(cfg, pl_mask):
    if cfg.dataset == 'cora':
        pl, explanations = answer_parser_cora(pl_mask)
    if cfg.dataset == 'wisconsin' or 'cornell' or 'texas':
        pl, explanations = answer_parser_webkb(cfg.dataset, cfg.llm.name, pl_mask)
    return pl, explanations