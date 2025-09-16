from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rti_sim.settings import load_settings
from rti_sim.pipelines import stage0_ingest, stage1_preprocess, stage2_embed, stage3_cluster
from rti_sim.pipelines.assign_cluster import run_assign

@pytest.fixture(autouse=True)
def patch_encoder(monkeypatch):
    import rti_sim.embedding.encoder as enc_mod
    class _DummyEncoder:
        def __init__(self, *a, **k): pass
        def encode(self, texts):
            out = []
            for t in texts:
                t = t or ""
                v = np.zeros(16, dtype=np.float32)
                v[0] = len(t)
                v[1] = sum(map(ord, t[:8]))
                n = np.linalg.norm(v) + 1e-12
                out.append(v / n)
            return np.vstack(out)
    monkeypatch.setattr(enc_mod, "TextEncoder", _DummyEncoder)
    yield

def test_assign_clusters(tmp_path: Path):
    # tiny setup
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "rti_master.csv").write_text(
        "AC_MSN,GTI_NUMBER,RTI_NUMBER,TITLE,WORKLOAD\n"
        "99999,GTI001,RTI001,Hydraulic Pump Failure,8\n"
        "99999,GTI001,RTI002,Hydraulic Pressure Drop,5\n", encoding="utf-8"
    )
    (raw / "rti_chapters.csv").write_text(
        "AC_MSN,GTI_NUMBER,RTI_NUMBER,CHAPTER_NUMBER,TITLE,ATTESTATION_TYPE\n"
        "99999,GTI001,RTI001,1,Introduction,QA\n"
        "99999,GTI001,RTI001,2,Failure Description,QA\n"
        "99999,GTI001,RTI002,1,Symptoms,QC\n", encoding="utf-8"
    )
    cfg_path = tmp_path / "configs" / "base.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(f"""
paths:
  rti_master: { (raw / 'rti_master.csv').as_posix() }
  rti_chapters: { (raw / 'rti_chapters.csv').as_posix() }
  interim_dir: { (tmp_path / 'data' / 'interim').as_posix() }/
  processed_dir: { (tmp_path / 'data' / 'processed').as_posix() }/
  out_dir: { (tmp_path / 'artifacts').as_posix() }/
filters:
  ac_msn_equals: "99999"
  drop_ac_msn_after_filter: true
text:
  generic_headings:
    fr: ["introduction"]
    en: ["introduction","overview"]
  meta_cols: ["workload","attestation_type"]
embeddings:
  model_name: "dummy"
  device: "cpu"
  normalize: true
  batch_size: 8
  e5_prefix: "passage: "
ann:
  backend: "sklearn"
  metric: "cosine"
cluster:
  algorithm: "kmeans"
  k: 2
  random_state: 0
  max_iter: 100
  assign:
    metric: "cosine"
    outlier_threshold: null
""", encoding="utf-8")

    cfg = load_settings(cfg_path)
    stage0_ingest.run(cfg)
    stage1_preprocess.run(cfg)
    stage2_embed.run(cfg)
    stage3_cluster.run(cfg)

    inp = tmp_path / "data" / "processed" / "views.parquet"
    df = run_assign(
        new_views_path=inp,
        out_dir=tmp_path / "artifacts",
        model_name="dummy",
        device="cpu",
        normalize=True,
        batch_size=8,
        metric="cosine",
        outlier_threshold=None,
    )
    assert {"rti_number","assigned_cluster","score","rank"}.issubset(df.columns)
    assert len(df) > 0
    assert (df["assigned_cluster"] >= 0).all()
