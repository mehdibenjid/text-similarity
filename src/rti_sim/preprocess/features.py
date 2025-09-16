from __future__ import annotations
import pandas as pd

def _canonicalize_generic(gen_cfg) -> set[str]:
    # gen_cfg peut être une liste (legacy) ou un dict {"fr":[..],"en":[..]}
    if isinstance(gen_cfg, dict):
        pool = []
        for v in gen_cfg.values():
            pool.extend(v or [])
    else:
        pool = gen_cfg or []
    # lower/strip
    return set([str(x).lower().strip() for x in pool if x])

def is_generic_heading(h: str, generic: set[str]) -> bool:
    h_l = h.lower().strip()
    return h_l in generic or len(h_l) < 2

def build_views(
    master: pd.DataFrame,
    chapters: pd.DataFrame,
    meta_cols: list[str],
    generic_headings: list[str],
) -> pd.DataFrame:
    """
    Construit 3 vues texte/canoniques par RTI :
      - title_text : titre master
      - chap_titles_text : concat des titres de chapitres ordonnés par chapter_number
      - meta_text : concat de meta (workload, attestation_type, etc.)
    """
    generic = _canonicalize_generic(generic_headings)

    base = master[["rti_number", "gti_number", "title"]].rename(columns={"title": "title_text"}).copy()

    chap = chapters[["rti_number", "chapter_number", "chapter_title"]].copy()
    chap = chap.sort_values(["rti_number", "chapter_number"])
    chap["chapter_title"] = chap["chapter_title"].fillna("").astype(str).str.strip()
    chap = chap[~chap["chapter_title"].map(lambda x: is_generic_heading(x, generic))]
    chap_agg = (
        chap.groupby("rti_number")["chapter_title"]
            .apply(lambda s: " || ".join([t for t in s if t]))
            .reset_index()
            .rename(columns={"chapter_title": "chap_titles_text"})
    )

    meta_cols = [c for c in (meta_cols or []) if c in set(master.columns) | set(chapters.columns)]
    meta_master = master[["rti_number"] + [c for c in meta_cols if c in master.columns]].copy()
    for c in [c for c in meta_cols if c in meta_master.columns]:
        meta_master[c] = meta_master[c].astype(str).fillna("").str.strip()

    chap_meta_cols = [c for c in meta_cols if c in chapters.columns]
    if chap_meta_cols:
        meta_ch = chapters[["rti_number"] + chap_meta_cols].copy()
        for c in chap_meta_cols:
            meta_ch[c] = meta_ch[c].astype(str).fillna("").str.strip()
        meta_ch = (
            meta_ch.groupby("rti_number")[chap_meta_cols]
            .agg(lambda s: " | ".join(sorted(set([v for v in s if v]))))
            .reset_index()
        )
        meta = pd.merge(meta_master, meta_ch, on="rti_number", how="outer")
    else:
        meta = meta_master

    keep_meta_cols = [c for c in meta.columns if c != "rti_number"]
    meta["meta_text"] = meta[keep_meta_cols].agg(
        lambda row: " | ".join([v for v in row if isinstance(v, str) and v]), axis=1
    ) if keep_meta_cols else ""
    meta = meta[["rti_number", "meta_text"]]

    out = base.merge(chap_agg, on="rti_number", how="left").merge(meta, on="rti_number", how="left")
    out["chap_titles_text"] = out["chap_titles_text"].fillna("")
    out["meta_text"] = out["meta_text"].fillna("")
    return out
