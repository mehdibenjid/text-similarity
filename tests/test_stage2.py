from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

# import path to src
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rti_sim.settings import load_settings
from rti_sim.pipelines import stage0_ingest, stage1_preprocess, stage2_embed
from rti_sim.pipelines.infer_knn import load_index_and_catalog, knn_assign

# --- fixtures: tiny data and config that uses sklearn ANN
@pytest.fixture
def tiny_raw(tmp_path: Path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    (raw / "rti_master.csv").write_text(
        "AC_MSN,GTI_NUMBER,RTI_NUMBER,TITLE,WORKLOAD\n"
        "99999,GTI001,RTI001,Hydraulic Pump Failure,8\n"
        "99999,GTI001,RTI002,Hydraulic Pressure Drop,5\n"
        "99999,GTI002,RTI003,Avionics Cooling Issue,2\n"
        "99999,GTI002,RTI004,Display Flickering,3\n", encoding="utf-8"
    )
    (raw / "rti_chapters.csv").write_text(
        "AC_MSN,GTI_NUMBER,RTI_NUMBER,CHAPTER_NUMBER,TITLE,ATTESTATION_TYPE\n"
        "99999,GTI001,RTI001,1,Introduction,QA\n"
        "99999,GTI001,RTI001,2,Failure Description,QA\n"
        "99999,GTI001,RTI002,1,Symptoms,QC\n"
        "99999,GTI002,RTI003,1,Overview,QA\n"
        "99999,GTI002,RTI004,1,Issue Report,QC\n", encoding="utf-8"
    )
    return tmp_path

@pytest.fixture
def cfg_stage2(tiny_raw: Path):
    cfg_dir = tiny_raw / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg_dir / "base.yaml"
    cfg.write_text(f"""
paths:
  rti_master: { (tiny_raw / 'data' / 'raw' / 'rti_master.csv').as_posix() }
  rti_chapters: { (tiny_raw / 'data' / 'raw' / 'rti_chapters.csv').as_posix() }
  interim_dir: { (tiny_raw / 'data' / 'interim').as_posix() }/
  processed_dir: { (tiny_raw / 'data' / 'processed').as_posix() }/
  out_dir: { (tiny_raw / 'artifacts').as_posix() }/

io:
  csv:
    encoding: "utf-8"
    sep: ","
    quotechar: '"'
    on_bad_lines: "skip"

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
    fr: ["introduction","résumé","conclusion","annexe","appendice","sommaire"]
    en: ["introduction","summary","conclusion","annex","appendix","table of contents","overview","background"]
  meta_cols: ["workload","attestation_type"]

embeddings:
  model_name: "dummy-not-used"   # we monkeypatch the encoder
  batch_size: 8
  device: "cpu"
  normalize: true
  weights:
    title_text: 2.0
    chap_titles_text: 1.0
    meta_text: 0.5
  e5_prefix: "passage: "

ann:
  backend: "sklearn"
  metric: "cosine"
""", encoding="utf-8")
    return cfg

# --- dummy encoder (deterministic, tiny, no downloads)
class _DummyEncoder:
    def __init__(self, model_name: str, device: str = "cpu", normalize: bool = True, batch_size: int = 32):
        self.normalize = normalize
    def encode(self, texts):
        rng = np.random.default_rng(42)
        # hash-ish: length + few char codes → 16-dim
        vecs = []
        for t in texts:
            t = t or ""
            arr = np.array([len(t), sum(map(ord, t[:10]))] + [0]*14, dtype=np.float32)
            noise = rng.normal(0, 0.01, size=16).astype(np.float32)
            v = arr + noise
            if self.normalize:
                n = np.linalg.norm(v) + 1e-12
                v = v / n
            vecs.append(v)
        return np.vstack(vecs)

@pytest.fixture(autouse=True)
def patch_encoder(monkeypatch):
    import rti_sim.embedding.encoder as enc_mod
    monkeypatch.setattr(enc_mod, "TextEncoder", _DummyEncoder)
    yield

def test_stage2_train_and_infer(cfg_stage2: Path):
    cfg = load_settings(cfg_stage2)

    # Stage 0 + 1
    stage0_ingest.run(cfg)
    meta1 = stage1_preprocess.run(cfg)
    views_p = Path(meta1["views_path"])
    assert views_p.exists()

    # Stage 2 (with dummy encoder + sklearn ANN)
    meta2 = stage2_embed.run(cfg)
    emb_p = Path(meta2["embeddings_path"])
    meta_p = Path(meta2["catalog_meta_path"])
    idx_path = meta2["index_path"]
    assert emb_p.exists() and meta_p.exists()
    assert idx_path and Path(idx_path).exists()

    # Inference against own catalog (sanity)
    df_new = pd.read_parquet(views_p)
    from rti_sim.embedding.encoder import TextEncoder  # patched to dummy
    from rti_sim.pipelines.infer_knn import load_index_and_catalog, knn_assign

    catalog_emb, catalog_meta, index_obj = load_index_and_catalog(Path(cfg.paths.out_dir), backend="sklearn")
    enc = TextEncoder("ignore", device="cpu", normalize=True, batch_size=8)
    assign = knn_assign(df_new, catalog_emb, catalog_meta, index_obj, enc, k=3, metric="cosine")

    assert set(assign.columns) == {"query_rti","candidate_rti","candidate_gti","score","rank"}
    # 10 queries * 3 neighbors
    assert len(assign) == len(df_new) * 3
