from core.config import cfg, update_cfg
from core.pipelines.lcgnn_ucscore_pipeline import run_lcgnn_with_ucscore

if __name__ == '__main__':
    runtime_cfg = update_cfg(cfg)
    run_lcgnn_with_ucscore(runtime_cfg)
