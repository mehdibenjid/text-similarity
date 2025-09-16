from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import pytest

# Assure l'import de src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rti_sim.settings import load_settings
from rti_sim.pipelines import stage1_preprocess

@pytest.fixture
def tiny_raw_bi(tmp_path: Path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Master: EN + FR titles
    (raw_dir / "rti_master.csv").write_text(
        "AC_MSN,GTI_NUMBER,RTI_NUMBER,TITLE,WORKLOAD\n"
        "99999,GTI001,RTI001,Hydraulic Pump Failure,8\n"      # EN
        "99999,GTI002,RTI002,Panne du système hydraulique,5\n" # FR
        , encoding="utf-8"
    )

    # Chapters: inclut des têtes génériques en FR et EN, qu'on doit filtrer
    (raw_dir / "rti_chapters.csv").write_text(
        "AC_MSN,GTI_NUMBER,RTI_NUMBER,CHAPTER_NUMBER,TITLE,ATTESTATION_TYPE\n"
        "99999,GTI001,RTI001,1,Introduction,QA\n"             # générique EN -> out
        "99999,GTI001,RTI001,2,Failure Description,QA\n"
        "99999,GTI002,RTI002,1,Résumé,QC\n"                   # générique FR -> out
        "99999,GTI002,RTI002,2,Description de la panne,QC\n"
        , encoding="utf-8"
    )
    return tmp_path

@pytest.fixture
def cfg_bi(tiny_raw_bi: Path):
    cfg_dir = tiny_raw_bi / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg_dir / "base.yaml"
    cfg.write_text(f"""
paths:
  rti_master: { (tiny_raw_bi / 'data' / 'raw' / 'rti_master.csv').as_posix() }
  rti_chapters: { (tiny_raw_bi / 'data' / 'raw' / 'rti_chapters.csv').as_posix() }
  interim_dir: { (tiny_raw_bi / 'data' / 'interim').as_posix() }/
  processed_dir: { (tiny_raw_bi / 'data' / 'processed').as_posix() }/
  out_dir: { (tiny_raw_bi / 'artifacts').as_posix() }/

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

run:
  run_id_prefix: "test"
""", encoding="utf-8")
    return cfg

def test_views_have_lang_and_filter_generics(cfg_bi: Path):
    cfg = load_settings(cfg_bi)
    meta = stage1_preprocess.run(cfg)

    v = pd.read_parquet(meta["views_path"])
    # colonnes attendues
    for col in ["rti_number","gti_number","title_text","chap_titles_text","meta_text","lang_title","lang_chapters","lang_rti"]:
        assert col in v.columns

    # RTI001 (EN): title EN, chapter generic "Introduction" filtré -> restant "Failure Description"
    r1 = v.set_index("rti_number").loc["RTI001"]
    assert r1["lang_title"] in (None, "en") or r1["lang_title"].startswith("en")
    assert "Introduction" not in r1["chap_titles_text"]
    assert "Failure Description" in r1["chap_titles_text"]

    # RTI002 (FR): title FR, chapter generic "Résumé" filtré -> restant "Description de la panne"
    r2 = v.set_index("rti_number").loc["RTI002"]
    assert r2["lang_title"] in (None, "fr") or r2["lang_title"].startswith("fr")
    assert "Résumé" not in r2["chap_titles_text"]
    assert "Description de la panne" in r2["chap_titles_text"]

    # lang_rti cohérent avec la majorité (titre dominant)
    assert r1["lang_rti"] in (None, "en", "fr")
    assert r2["lang_rti"] in (None, "en", "fr")
