from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from rti_sim.embedding.textcraft import build_embedding_text
from rti_sim.embedding import encoder as enc_mod
from rti_sim.logging import get_logger

log = get_logger()

def load_centroids(out_dir: Path):
    C = np.load(out_dir / "cluster_centroids.npy")  # (k, d) normalisés
    meta = json.load(open(out_dir / "cluster_meta.json", "r", encoding="utf-8"))
    return C, meta

def encode_views(df_new: pd.DataFrame, model_name: str, device: str, normalize: bool, batch_size: int) -> np.ndarray:
    Encoder = enc_mod.TextEncoder  # lazy-resolved for tests
    texts = build_embedding_text(df_new).tolist()
    enc = Encoder(model_name=model_name, device=device, normalize=normalize, batch_size=batch_size)
    return enc.encode(texts)

def assign_to_clusters(
    df_new: pd.DataFrame,
    centroids: np.ndarray,
    q_emb: np.ndarray,
    metric: str = "cosine",
    outlier_threshold: float | None = None,
) -> pd.DataFrame:
    # embeddings normalisés → dot = cosine
    S = q_emb @ centroids.T  # (nq, k)
    top = S.argmax(axis=1)
    top_sim = S[np.arange(q_emb.shape[0]), top]

    cluster_id = top.astype(int)
    if outlier_threshold is not None:
        cluster_id = np.where(top_sim >= float(outlier_threshold), cluster_id, -1)

    out = pd.DataFrame({
        "rti_number": df_new.get("rti_number", pd.Series(range(len(df_new)))).astype(str),
        "assigned_cluster": cluster_id.astype(int),
        "score": top_sim.astype(np.float32),
        "rank": 1,
    })
    return out

def run_assign(
    new_views_path: Path,
    out_dir: Path,
    model_name: str,
    device: str,
    normalize: bool,
    batch_size: int,
    metric: str = "cosine",
    outlier_threshold: float | None = None,
) -> pd.DataFrame:
    df_new = pd.read_parquet(new_views_path) if new_views_path.suffix.lower() != ".csv" else pd.read_csv(new_views_path, dtype=str)
    C, _ = load_centroids(out_dir)
    Q = encode_views(df_new, model_name=model_name, device=device, normalize=normalize, batch_size=batch_size)
    return assign_to_clusters(df_new, C, Q, metric=metric, outlier_threshold=outlier_threshold)
