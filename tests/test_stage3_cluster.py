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

@pytest.fixture
def tiny_env(tmp_path: Path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "rti_master.csv").write_text(
        "AC_MSN,GTI_NUMBER,RTI_NUMBER,TITLE,WORKLOAD\n"
        "99999,GTI001,RTI001,Hydraulic Pump Failure,8\n"
        "99999,GTI001,RTI002,Hydraulic Pressure Drop,5\n"
        "99999,GTI002,RTI003,Avionics Cooling Issue,2\n"
        "99999,GTI002,RTI004,Display Flickering,3\n",
        encoding="utf-8"
    )
    (raw / "rti_chapters.csv").write_text(
        "AC_MSN,GTI_NUMBER,RTI_NUMBER,CHAPTER_NUMBER,TITLE,ATTESTATION_TYPE\n"
        "99999,GTI001,RTI001,1,Introduction,QA\n"
        "99999,GTI001,RTI001,2,Failure Description,QA\n"
        "99999,GTI001,RTI002,1,Symptoms,QC\n"
        "99999,GTI002,RTI003,1,Overview,QA\n"
        "99999,GTI002,RTI004,1,Issue Report,QC\n",
        encoding="utf-8"
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
columns:
  master_keep: ["AC_MSN","gti_number","rti_number","title","workload"]
  chapters_keep: ["AC_MSN","gti_number","rti_number","chapter_number","chapter_title","attestation_type"]
  aliases_master:
    AC_MSN: AC_MSN
    GTI_NUMBER: gti_number
    RTI_NUMBER: rti_number
    TITLE: title
    WORKLOAD: workload
  aliases_chapters:
    AC_MSN: AC_MSN
    GTI_NUMBER: gti_number
    RTI_NUMBER: rti_number
    CHAPTER_NUMBER: chapter_number
    TITLE: chapter_title
    ATTESTATION_TYPE: attestation_type
filters:
  ac_msn_equals: "99999"
  drop_ac_msn_after_filter: true
text:
  generic_headings:
    fr: ["introduction"]
    en: ["introduction","overview"]
  meta_cols: ["workload","attestation_type"]
embeddings:
  model_name: "intfloat/multilingual-e5-base"
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
""", encoding="utf-8")
    return cfg_path

def test_stage3_cluster_end_to_end(tiny_env: Path, monkeypatch):
    # Monkeypatch encoder to a tiny deterministic dummy to avoid downloading models
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

    cfg = load_settings(tiny_env)
    stage0_ingest.run(cfg)
    stage1_preprocess.run(cfg)
    stage2_embed.run(cfg)

    meta3 = stage3_cluster.run(cfg)
    assert "cluster_assignments_path" in meta3
    assign = pd.read_parquet(meta3["cluster_assignments_path"])
    assert {"rti_number","gti_number","cluster_id","sim_to_centroid"}.issubset(assign.columns)
    assert (assign["cluster_id"] >= 0).all()
