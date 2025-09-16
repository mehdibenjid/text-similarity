from __future__ import annotations
from pathlib import Path
import pandas as pd
from rti_sim.settings import Settings
from rti_sim.logging import get_logger
from rti_sim.io.load import load_master, load_chapters
from rti_sim.preprocess.clean import clean_columns
from rti_sim.preprocess.features import build_views
from rti_sim.preprocess.filters import filter_ac_msn
from rti_sim.preprocess.lang_detect import detect_lang, majority_lang

log = get_logger()

def _cfg_dict(cfg: Settings) -> dict:
    return cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()

def run(cfg: Settings) -> dict:
    interim = Path(cfg.paths.interim_dir)
    processed = Path(cfg.paths.processed_dir)
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    cfgd = _cfg_dict(cfg)
    csv_opts = (cfgd.get("io", {}) or {}).get("csv", {}) or {}

    cols_cfg = cfgd.get("columns", {}) or {}
    master_keep = cols_cfg.get("master_keep", ["AC_MSN","gti_number","rti_number","title","workload"])
    chapters_keep = cols_cfg.get("chapters_keep", ["AC_MSN","gti_number","rti_number","chapter_number","chapter_title","attestation_type"])
    aliases_master = cols_cfg.get("aliases_master", {}) or {}
    aliases_chapters = cols_cfg.get("aliases_chapters", {}) or {}

    filt = cfgd.get("filters", {}) or {}
    ac_msn_value = str(filt.get("ac_msn_equals", "99999"))
    drop_ac_msn_after = bool(filt.get("drop_ac_msn_after_filter", True))

    master = load_master(cfg.paths.rti_master, csv_opts, master_keep, aliases_master)
    chapters = load_chapters(cfg.paths.rti_chapters, csv_opts, chapters_keep, aliases_chapters)
    ##"##"
    log.info(master.head().to_string())
    log.info(chapters.head().to_string())
    ####
    log.info(f"[stage1] master columns -> {list(master.columns)}")
    log.info(f"[stage1] chapters columns -> {list(chapters.columns)}")

    master = clean_columns(master, [c for c in ["title"] if c in master.columns])
    chapters = clean_columns(chapters, [c for c in ["chapter_title","attestation_type"] if c in chapters.columns])

    # caster proprement workload si présent
    if "workload" in master.columns:
        master["workload"] = pd.to_numeric(master["workload"], errors="coerce")

    # Filtre AC_MSN
    before_m, before_c = len(master), len(chapters)
    master = filter_ac_msn(master, ac_msn_value, drop_after=drop_ac_msn_after)
    chapters = filter_ac_msn(chapters, ac_msn_value, drop_after=drop_ac_msn_after)
    log.info(f"[stage1] AC_MSN=={ac_msn_value}: master {before_m} -> {len(master)}, chapters {before_c} -> {len(chapters)}")

    # Sauvegarde cleaned
    master_path = interim / "clean_rti_master.parquet"
    ch_path = interim / "clean_rti_chapters.parquet"
    master.to_parquet(master_path, index=False)
    chapters.to_parquet(ch_path, index=False)

    # Vues texte
    views = build_views(
        master=master,
        chapters=chapters,
        meta_cols=cfgd.get("text", {}).get("meta_cols", ["workload","attestation_type"]),
        generic_headings=cfgd.get("text", {}).get("generic_headings", []),
    )
    views["lang_title"] = views["title_text"].map(detect_lang)
    views["lang_chapters"] = views["chap_titles_text"].map(detect_lang)
    views["lang_rti"] = [
        majority_lang(t, c) for t, c in zip(views["lang_title"], views["lang_chapters"])
    ]
    views_path = processed / "views.parquet"
    views.to_parquet(views_path, index=False)

    log.info(f"[stage1] Cleaned + views written: {views_path} ({len(views)} rows)")
    return {
        "master_clean_path": str(master_path),
        "chapters_clean_path": str(ch_path),
        "views_path": str(views_path),
        "rows_master": len(master),
        "rows_chapters": len(chapters),
        "rows_views": len(views),
    }
