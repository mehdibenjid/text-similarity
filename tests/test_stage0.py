# tests/test_stage0.py
from __future__ import annotations
from pathlib import Path
from rti_sim.settings import load_settings
from rti_sim.pipelines import stage0_ingest

def test_stage0_creates_outdir(tiny_config):
    cfg = load_settings(tiny_config)
    meta = stage0_ingest.run(cfg)
    assert "run_id" in meta and meta["run_id"].startswith("test-")
    out_dir = Path(meta["out_dir"])
    assert out_dir.exists() and out_dir.is_dir()
