from __future__ import annotations
from pathlib import Path
from datetime import datetime
from rti_sim.settings import Settings
from rti_sim.logging import get_logger

log = get_logger()

def run(cfg: Settings) -> dict:
    """
    Stage 0: verify config, ensure out dir, warn about inputs, emit run_id.
    No data is read or written (except creating artifacts/).
    """
    out_dir = Path(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, p in [("rti_master", cfg.paths.rti_master), ("rti_chapters", cfg.paths.rti_chapters)]:
        path = Path(p)
        if not path.exists():
            log.warning(f"[stage0] Input path missing: {name} -> {path}")

    run_id = f"{cfg.run.run_id_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log.info(f"[stage0] Initialized run_id={run_id}; artifacts -> {out_dir}")
    return {"run_id": run_id, "out_dir": str(out_dir)}
