from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rti_sim.settings import Settings
from rti_sim.logging import get_logger

log = get_logger()

def _cfg_dict(cfg: Settings) -> dict:
    return cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()

def _load_artifacts(paths: dict, out_dir: Path):
    views_p = Path(paths["processed_dir"]) / "views.parquet"
    assign_p = out_dir / "cluster_assignments.parquet"
    centroids_p = out_dir / "cluster_centroids.npy"
    for p in (views_p, assign_p, centroids_p):
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")
    views = pd.read_parquet(views_p)
    assign = pd.read_parquet(assign_p)
    C = np.load(centroids_p)
    return views, assign, C

def _make_corpus(df: pd.DataFrame) -> pd.Series:
    # simple, robust corpus from Phase 1 columns
    title = df.get("title_text", "").fillna("").astype(str)
    chaps = df.get("chap_titles_text", "").fillna("").astype(str)
    # lightweight meta
    meta = df.get("meta_text", "").fillna("").astype(str)
    corpus = (title + " \n " + chaps + " \n " + meta).str.lower()
    return corpus

def _tfidf_terms_per_cluster(corpus: pd.Series, labels: pd.Series, top_k: int = 8) -> dict:
    # fit one vectorizer to whole corpus, then rank terms per cluster by avg tf-idf
    vect = TfidfVectorizer(
        strip_accents="unicode",
        min_df=1,
        max_df=0.9,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w\-]{2,}\b",
    )
    X = vect.fit_transform(corpus.tolist())  # sparse (N, V)
    terms = np.array(vect.get_feature_names_out())
    result: dict[int, list[str]] = {}
    for cid, idx in labels.groupby(labels).groups.items():
        sub = X[idx]  # rows in this cluster
        # mean tf-idf across rows
        mean_tfidf = np.asarray(sub.mean(axis=0)).ravel()
        top_idx = np.argsort(-mean_tfidf)[:top_k]
        result[int(cid)] = terms[top_idx].tolist()
    return result

def _exemplars(df_assign: pd.DataFrame, k_top: int = 3) -> dict:
    # pick the k samples with highest sim_to_centroid per cluster
    ex = {}
    for cid, grp in df_assign.groupby("cluster_id", sort=True):
        top = grp.sort_values("sim_to_centroid", ascending=False).head(k_top)
        ex[int(cid)] = top["rti_number"].astype(str).tolist()
    return ex

def run(cfg: Settings) -> dict:
    cfgd = _cfg_dict(cfg)
    out_dir = Path(cfg.paths.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    views, assign, C = _load_artifacts(cfgd["paths"], out_dir)

    text_cols = [c for c in ["rti_number", "title_text", "chap_titles_text", "meta_text"] if c in views.columns]
    views_text = views[text_cols].copy()

    df = assign.merge(views_text, on="rti_number", how="left")

    # sécurité : s'assurer qu'on a bien une colonne gti_number
    if "gti_number" not in df.columns:
        # cas où un autre code aurait déjà provoqué _x/_y
        if "gti_number_x" in df.columns or "gti_number_y" in df.columns:
            gx = df["gti_number_x"] if "gti_number_x" in df.columns else None
            gy = df["gti_number_y"] if "gti_number_y" in df.columns else None
            if gx is not None and gy is not None:
                df["gti_number"] = gx.fillna(gy).astype(str)
            elif gx is not None:
                df["gti_number"] = gx.astype(str)
            elif gy is not None:
                df["gti_number"] = gy.astype(str)
            # nettoyage
            for col in ("gti_number_x", "gti_number_y"):
                if col in df.columns:
                    df.drop(columns=col, inplace=True)
        else:
            # rien à merger → message clair pour debug
            raise KeyError("gti_number missing after merge; expected in cluster_assignments.parquet")

    # GTI distribution per cluster
    gti_counts = (
        df.groupby(["cluster_id", "gti_number"], dropna=False)
          .size()
          .reset_index(name="count")
    )
    gti_counts["pct"] = (
        gti_counts["count"] / gti_counts.groupby("cluster_id")["count"].transform("sum")
    )

    # purity: majority GTI share per cluster
    maj = (
        gti_counts.sort_values(["cluster_id", "pct"], ascending=[True, False])
                  .groupby("cluster_id")
                  .head(1)
                  .rename(columns={"gti_number":"major_gti", "pct":"purity"})
    )[["cluster_id","major_gti","purity"]]

    # size and avg sim
    size_sim = (
        df.groupby("cluster_id")
          .agg(size=("rti_number","count"), avg_sim=("sim_to_centroid","mean"))
          .reset_index()
    )

    # top terms (title + chapters + meta)
    corpus = _make_corpus(df)
    top_terms = _tfidf_terms_per_cluster(corpus, df["cluster_id"])
    exemplars = _exemplars(df, k_top=3)

    # assemble summary
    summary = (
        size_sim.merge(maj, on="cluster_id", how="left")
                .sort_values("cluster_id")
                .reset_index(drop=True)
    )
    # attach top terms/exemplars as JSON strings for convenience in parquet
    summary["top_terms"] = summary["cluster_id"].map(lambda cid: json.dumps(top_terms.get(int(cid), []), ensure_ascii=False))
    summary["exemplars"] = summary["cluster_id"].map(lambda cid: json.dumps(exemplars.get(int(cid), []), ensure_ascii=False))

    # save artifacts
    out_summary = out_dir / "cluster_summary.parquet"
    out_gti = out_dir / "cluster_gti_distribution.parquet"
    out_terms = out_dir / "cluster_label_terms.json"
    out_report = out_dir / "cluster_report.json"

    summary.to_parquet(out_summary, index=False)
    gti_counts.to_parquet(out_gti, index=False)
    with open(out_terms, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in top_terms.items()}, f, ensure_ascii=False, indent=2)

    # quick global summary
    report = {
        "clusters": int(summary.shape[0]),
        "total_rti": int(df.shape[0]),
        "avg_cluster_size": float(summary["size"].mean()) if len(summary) else 0.0,
        "mean_purity": float(summary["purity"].mean()) if "purity" in summary else 0.0,
        "mean_avg_sim": float(summary["avg_sim"].mean()) if "avg_sim" in summary else 0.0,
    }
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info(f"[stage4] Wrote {out_summary.name}, {out_gti.name}, {out_terms.name}, {out_report.name}")
    return {
        "cluster_summary_path": str(out_summary),
        "cluster_gti_distribution_path": str(out_gti),
        "cluster_terms_path": str(out_terms),
        "cluster_report_path": str(out_report),
        **report,
    }
