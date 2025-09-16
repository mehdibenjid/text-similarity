from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from rti_sim.settings import Settings
from rti_sim.logging import get_logger
from rti_sim.embedding.textcraft import build_embedding_text
from rti_sim.embedding.encoder import TextEncoder
from rti_sim.index.ann import build_index

log = get_logger()

def _cfg_dict(cfg: Settings) -> dict:
    return cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()

def run(cfg: Settings) -> dict:
    cfgd = _cfg_dict(cfg)
    processed = Path(cfg.paths.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    outdir = Path(cfg.paths.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    views_path = processed / "views.parquet"
    if not views_path.exists():
        raise FileNotFoundError(f"Missing {views_path}. Run Stage 1 first.")

    df = pd.read_parquet(views_path)
    log.info(f"[stage2] Loaded views: {views_path} ({len(df)})")

    emb_cfg = cfgd.get("embeddings", {}) or {}
    w = emb_cfg.get("weights", {}) or {}
    text = build_embedding_text(
        df,
        w_title=float(w.get("title_text", 2.0)),
        w_chaps=float(w.get("chap_titles_text", 1.0)),
        w_meta=float(w.get("meta_text", 0.5)),
        e5_prefix=str(emb_cfg.get("e5_prefix", "passage: ")),
    )

    encoder = TextEncoder(
        model_name=str(emb_cfg.get("model_name", "intfloat/multilingual-e5-base")),
        device=str(emb_cfg.get("device", "auto")),
        normalize=bool(emb_cfg.get("normalize", True)),
        batch_size=int(emb_cfg.get("batch_size", 64)),
    )

    emb = encoder.encode(text.tolist())  # (N, d) float32
    log.info(f"[stage2] Encoded embeddings: shape={emb.shape}")

    # Save artifacts
    emb_path = outdir / "catalog_embeddings.npy"
    np.save(emb_path, emb)
    meta_path = outdir / "catalog_meta.parquet"
    df[["rti_number", "gti_number"]].to_parquet(meta_path, index=False)

    # Build ANN index
    ann_cfg = cfgd.get("ann", {}) or {}
    backend = ann_cfg.get("backend", "faiss")
    metric = ann_cfg.get("metric", "cosine")
    faiss_use_ivf = bool(ann_cfg.get("faiss", {}).get("use_ivf", False))
    nlist = int(ann_cfg.get("faiss", {}).get("nlist", 1024))

    index = build_index(emb, backend=backend, metric=metric, faiss_use_ivf=faiss_use_ivf, nlist=nlist)

    # Persist index
    idx_path = None
    if backend == "faiss":
        try:
            import faiss  # type: ignore
            faiss.write_index(index, str(outdir / "faiss.index"))
            idx_path = outdir / "faiss.index"
        except Exception as e:
            log.warning(f"[stage2] Could not write FAISS index ({e}); falling back to sklearn.")
            # build sklearn index as fallback
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=10, algorithm="brute", metric=metric)
            nn.fit(emb)
            from joblib import dump
            idx_path = outdir / "sklearn_nn.joblib"
            dump(nn, idx_path)
    else:
        from joblib import dump
        idx_path = outdir / "sklearn_nn.joblib"
        dump(index, idx_path)


    return {
        "views_path": str(views_path),
        "embeddings_path": str(emb_path),
        "catalog_meta_path": str(meta_path),
        "index_path": str(idx_path) if idx_path else None,
        "n": int(emb.shape[0]),
        "d": int(emb.shape[1]),
        "backend": backend,
        "metric": metric,
    }
