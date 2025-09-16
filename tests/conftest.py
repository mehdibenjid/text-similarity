# tests/conftest.py
from __future__ import annotations
import sys
from pathlib import Path
import pytest

# Assure que "src" est importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

@pytest.fixture
def tiny_raw(tmp_path: Path):
    """Crée de petits CSV bruts (masters + chapters) avec et sans AC_MSN=99999."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    master_csv = raw_dir / "rti_master.csv"
    master_csv.write_text(
        "AC_MSN,GTI_NUMBER,RTI_NUMBER,TITLE,WORKLOAD\n"
        "99999,GTI001,RTI001,Hydraulic Pump Failure,8\n"
        "99999,GTI001,RTI002,Hydraulic Pressure Drop,5\n"
        "12345,GTI999,RTI999,Bad Row Should Be Filtered,3\n", encoding="utf-8"
    )

    chapters_csv = raw_dir / "rti_chapters.csv"
    chapters_csv.write_text(
        "AC_MSN,GTI_NUMBER,RTI_NUMBER,CHAPTER_NUMBER,TITLE,ATTESTATION_TYPE\n"
        "99999,GTI001,RTI001,1,Introduction,QA\n"     # générique -> doit être filtré
        "99999,GTI001,RTI001,2,Failure Description,QA\n"
        "99999,GTI001,RTI002,1,Symptoms,QC\n"
        "12345,GTI999,RTI999,1,Noise,QC\n", encoding="utf-8"
    )

    return {
        "raw_dir": raw_dir,
        "master_csv": master_csv,
        "chapters_csv": chapters_csv,
        "tmp_root": tmp_path,
    }

@pytest.fixture
def tiny_config(tiny_raw):
    """Construit un fichier YAML de config minimal pointant vers les CSV temporaires."""
    tmp_root = tiny_raw["tmp_root"]
    cfg_dir = tmp_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # NB: cette config suppose que tu as bien implémenté Settings(io, columns, filters, text)
    cfg_yaml = cfg_dir / "base.yaml"
    cfg_yaml.write_text(f"""
paths:
  rti_master: {tiny_raw['master_csv'].as_posix()}
  rti_chapters: {tiny_raw['chapters_csv'].as_posix()}
  interim_dir: { (tmp_root / 'data' / 'interim').as_posix() }/
  processed_dir: { (tmp_root / 'data' / 'processed').as_posix() }/
  out_dir: { (tmp_root / 'artifacts').as_posix() }/

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
  generic_headings: ["introduction","summary","conclusion","annex","appendix","table of contents"]
  meta_cols: ["workload","attestation_type"]

run:
  run_id_prefix: "test"
""", encoding="utf-8")

    return cfg_yaml
