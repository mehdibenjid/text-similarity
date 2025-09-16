from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from rti_sim.settings import Settings
from rti_sim.logging import get_logger

log = get_logger()

def _cfg_dict(cfg: Settings) -> dict:
    return cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()

def _fit_kmeans(X: np.ndarray, k: int, random_state: int, max_iter: int, minibatch: bool, batch_size: int):
    if minibatch:
        model = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=batch_size, max_iter=max_iter, n_init="auto")
    else:
        model = KMeans(n_clusters=k, random_state=random_state, max_iter=max_iter, n_init="auto")
    model.fit(X)
    return model

def _centroids_unit(model, X: np.ndarray) -> np.ndarray:
    # Les centroids de KMeans ne sont pas forcément normalisés. On normalise pour cosine.
    C = model.cluster_centers_.astype(np.float32)
    n = np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    return C / n

def run(cfg: Settings) -> dict:
    cfgd = _cfg_dict(cfg)
    outdir = Path(cfg.paths.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    emb_path = outdir / "catalog_embeddings.npy"
    meta_path = outdir / "catalog_meta.parquet"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Missing embeddings or catalog meta. Run Phase 2 first.")

    X = np.load(emb_path)                     # (N, d), float32 normalisé
    meta = pd.read_parquet(meta_path)         # rti_number, gti_number
    assert X.shape[0] == len(meta), "Embeddings and meta row count mismatch."

    cl_cfg = cfgd.get("cluster", {}) or {}
    algo = str(cl_cfg.get("algorithm", "kmeans")).lower()
    k = int(cl_cfg.get("k", 8))
    random_state = int(cl_cfg.get("random_state", 42))
    max_iter = int(cl_cfg.get("max_iter", 300))
    use_mb = (algo == "minibatch")
    batch_size = int((cl_cfg.get("minibatch", {}) or {}).get("batch_size", 2048))

    log.info(f"[stage3] Clustering algo={algo} k={k}")
    model = _fit_kmeans(X, k=k, random_state=random_state, max_iter=max_iter, minibatch=use_mb, batch_size=batch_size)

    labels = model.labels_.astype(int)
    C = _centroids_unit(model, X)  # (k, d), normalisés

    # Qualité (sur cosine → on peut donner X tel quel, silhouette supporte 'euclidean' seulement; on approx avec euclidienne)
    sil = float(silhouette_score(X, labels, metric="euclidean")) if len(set(labels)) > 1 else -1.0
    dbi = float(davies_bouldin_score(X, labels)) if len(set(labels)) > 1 else -1.0

    # distance à centroid = 1 - (x dot c) avec embeddings normalisés
    sims = X @ C.T                # (N, k)
    best = sims.argmax(axis=1)    # labels devraient matcher
    best_sim = sims[np.arange(X.shape[0]), best]
    assert np.all(best == labels), "Model.labels_ and cosine argmax disagree; check normalization."

    assign = meta.copy()
    assign["cluster_id"] = labels
    assign["sim_to_centroid"] = best_sim.astype(np.float32)

    # Sauvegardes
    assign_path = outdir / "cluster_assignments.parquet"
    centroids_path = outdir / "cluster_centroids.npy"
    meta_json = {
        "algorithm": algo,
        "k": k,
        "random_state": random_state,
        "max_iter": max_iter,
        "silhouette_euclid": sil,
        "davies_bouldin": dbi,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
    }
    with open(outdir / "cluster_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)

    assign.to_parquet(assign_path, index=False)
    np.save(centroids_path, C)

    log.info(f"[stage3] Clusters saved: {assign_path}, centroids: {centroids_path}, sil={sil:.3f}, dbi={dbi:.3f}")
    return {
        "cluster_assignments_path": str(assign_path),
        "cluster_centroids_path": str(centroids_path),
        "cluster_meta_path": str(outdir / 'cluster_meta.json'),
        "silhouette": sil,
        "davies_bouldin": dbi,
        "k": k,
    }
