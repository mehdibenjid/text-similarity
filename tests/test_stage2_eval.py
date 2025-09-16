from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rti_sim.settings import load_settings
from rti_sim.pipelines import stage0_ingest, stage1_preprocess, stage2_embed
from rti_sim.pipelines.eval_quality import evaluate

class _DummyEncoder:
    def __init__(self, model_name: str, device: str = "cpu", normalize: bool = True, batch_size: int = 32):
        self.normalize = normalize
    def encode(self, texts):
        rng = np.random.default_rng(123)
        vecs = []
        for t in texts:
            t = t or ""
            arr = np.array([len(t), sum(map(ord, t[:10]))] + [0]*14, dtype=np.float32)
            if self.normalize:
                n = np.linalg.norm(arr) + 1e-12
                arr = arr / n
            vecs.append(arr)
        return np.vstack(vecs)

@pytest.fixture(autouse=True)
def patch_encoder(monkeypatch):
    import rti_sim.embedding.encoder as enc_mod
    monkeypatch.setattr(enc_mod, "TextEncoder", _DummyEncoder)
    yield

def test_eval_basic(tmp_path: Path):
    # minimal config reused from previous test, but inline for isolation
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

    cfg = tmp_path / "configs" / "base.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(f"""
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
  model_name: "dummy"
  batch_size: 8
  device: "cpu"
  normalize: true
  weights: {{}}
  e5_prefix: "passage: "

ann:
  backend: "sklearn"
  metric: "cosine"
""", encoding="utf-8")

    from rti_sim.settings import load_settings
    cfg_obj = load_settings(cfg)

    stage0_ingest.run(cfg_obj)
    stage1_preprocess.run(cfg_obj)
    stage2_embed.run(cfg_obj)

    rep = evaluate(
        views_path=(tmp_path / "data" / "processed" / "views.parquet"),
        out_dir=(tmp_path / "artifacts"),
        model_name="dummy",
        device="cpu",
        normalize=True,
        batch_size=8,
        backend="sklearn",
        metric="cosine",
        k=2,
    )
    assert "p@1" in rep and "by_lang" in rep
    assert rep["n"] == 2
