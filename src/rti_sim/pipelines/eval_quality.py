from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from rti_sim.embedding.encoder import TextEncoder
from rti_sim.embedding.textcraft import build_embedding_text
from rti_sim.index.ann import query_index
from rti_sim.pipelines.infer_knn import load_index_and_catalog
from rti_sim.logging import get_logger

log = get_logger()

def _precision_at_k(rels: np.ndarray, k: int) -> float:
    if rels.size == 0:
        return 0.0
    k = min(k, rels.shape[1])
    return float(rels[:, :k].mean())

def _map_at_k(rels: np.ndarray, k: int) -> float:
    N, K = rels.shape
    k = min(k, K)
    ap = []
    for i in range(N):
        hits = 0
        precs = []
        for r in range(k):
            if rels[i, r]:
                hits += 1
                precs.append(hits / (r + 1))
        ap.append(np.mean(precs) if precs else 0.0)
    return float(np.mean(ap))

def evaluate(
    views_path: Path,
    out_dir: Path,
    model_name: str,
    device: str,
    normalize: bool,
    batch_size: int,
    backend: str,
    metric: str,
    k: int = 10,
) -> dict:
    df = pd.read_parquet(views_path)
    req = {"rti_number","gti_number","title_text","chap_titles_text","meta_text"}
    if not req.issubset(df.columns):
        raise ValueError(f"views.parquet missing columns: {sorted(req - set(df.columns))}")

    catalog_emb, catalog_meta, index_obj = load_index_and_catalog(out_dir, backend=backend)

    encoder = TextEncoder(model_name=model_name, device=device, normalize=normalize, batch_size=batch_size)
    q_texts = build_embedding_text(df).tolist()
    Q = encoder.encode(q_texts)

    if index_obj is not None:
        idx, _score = query_index(index_obj, Q, k=k+1, metric=metric)
    else:
        sim = Q @ catalog_emb.T  # normalized → cosine
        idx = np.argsort(-sim, axis=1)[:, :k+1]

    cat_rti = catalog_meta["rti_number"].tolist()
    gold_gti = df["gti_number"].tolist()

    filtered_rel = []
    for i in range(len(df)):
        row = [j for j in idx[i] if cat_rti[j] != df.iloc[i]["rti_number"]][:k]
        rel = [1 if catalog_meta.iloc[j]["gti_number"] == gold_gti[i] else 0 for j in row]
        filtered_rel.append(rel)

    rels = np.array(filtered_rel, dtype=int)
    p_at = {f"p@{kk}": _precision_at_k(rels, kk) for kk in (1,3,5,10) if kk <= k}
    mapk = {f"map@{kk}": _map_at_k(rels, kk) for kk in (3,5,10) if kk <= k}

    strat = {}
    if "lang_rti" in df.columns:
        for lang in ["fr","en"]:
            mask = (df["lang_rti"] == lang)
            if mask.any():
                sub = rels[mask.values]
                strat[lang] = {
                    **{f"p@{kk}": _precision_at_k(sub, kk) for kk in (1,3,5,10) if kk <= k},
                    **{f"map@{kk}": _map_at_k(sub, kk) for kk in (3,5,10) if kk <= k},
                    "n": int(mask.sum()),
                }

    return {"n": int(len(df)), **p_at, **mapk, "by_lang": strat}
