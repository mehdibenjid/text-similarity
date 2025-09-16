# tests/test_stage1.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from rti_sim.settings import load_settings
from rti_sim.pipelines import stage1_preprocess

def test_stage1_end_to_end(tiny_config):
    cfg = load_settings(tiny_config)
    meta = stage1_preprocess.run(cfg)

    # fichiers produits
    master_p = Path(meta["master_clean_path"])
    ch_p = Path(meta["chapters_clean_path"])
    views_p = Path(meta["views_path"])
    assert master_p.exists() and ch_p.exists() and views_p.exists()

    # master: colonnes canoniques + filtrage AC_MSN
    m = pd.read_parquet(master_p)
    assert set(["gti_number","rti_number","title","workload"]).issubset(m.columns)
    # AC_MSN supprimé après filtre (cf. config drop=true)
    assert "AC_MSN" not in m.columns
    # la ligne AC_MSN=12345 a été filtrée
    assert not ((m["rti_number"] == "RTI999").any())

    # chapters: colonnes canoniques + filtrage AC_MSN
    c = pd.read_parquet(ch_p)
    assert set(["gti_number","rti_number","chapter_number","chapter_title","attestation_type"]).issubset(c.columns)
    assert "AC_MSN" not in c.columns
    assert not ((c["rti_number"] == "RTI999").any())

    # views: contenu attendu
    v = pd.read_parquet(views_p)
    expected_cols = ["rti_number","gti_number","title_text","chap_titles_text","meta_text"]
    # autoriser des colonnes additionnelles (langues, etc.)
    assert set(expected_cols).issubset(v.columns)

    # 2 RTI restants (RTI001, RTI002)
    assert set(v["rti_number"]) == {"RTI001","RTI002"}

    # RTI001 : chapitres -> "Failure Description" uniquement (Introduction filtré)
    row1 = v.set_index("rti_number").loc["RTI001"]
    assert row1["title_text"] == "Hydraulic Pump Failure"
    assert row1["chap_titles_text"] == "Failure Description"
    # meta_text contient workload "8" (stringifié) et attestation "QA"
    assert "8" in row1["meta_text"]
    assert "QA" in row1["meta_text"]

    # RTI002 : un chapitre "Symptoms", attestation "QC", workload "5"
    row2 = v.set_index("rti_number").loc["RTI002"]
    assert row2["title_text"] == "Hydraulic Pressure Drop"
    assert row2["chap_titles_text"] == "Symptoms"
    assert "5" in row2["meta_text"] and "QC" in row2["meta_text"]
