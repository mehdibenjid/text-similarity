from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from rti_sim.logging import get_logger
from rti_sim.embedding.textcraft import build_embedding_text
from rti_sim.embedding.encoder import TextEncoder
from rti_sim.index.ann import query_index

log = get_logger()

def load_index_and_catalog(out_dir: Path, backend: str):
    emb = np.load(out_dir / "catalog_embeddings.npy")
    meta = pd.read_parquet(out_dir / "catalog_meta.parquet")

    if backend == "faiss":
        try:
            import faiss  # type: ignore
            index = faiss.read_index(str(out_dir / "faiss.index"))
        except Exception as e:
            log.warning(f"[infer] FAISS index not available ({e}); switching to brute-force.")
            index = None
    else:
        from joblib import load
        index = load(out_dir / "sklearn_nn.joblib")
    return emb, meta, index

def knn_assign(
    df_new: pd.DataFrame,
    catalog_emb: np.ndarray,
    catalog_meta: pd.DataFrame,
    index_obj: object | None,
    encoder: TextEncoder,
    k: int = 5,
    metric: str = "cosine",
) -> pd.DataFrame:
    texts = build_embedding_text(df_new).tolist()
    q = encoder.encode(texts)

    if index_obj is not None:
        idx, score = query_index(index_obj, q, k=k, metric=metric)
    else:
        # brute-force: cosine sim = dot if normalized
        sim = q @ catalog_emb.T
        idx = np.argsort(-sim, axis=1)[:, :k]
        score = np.take_along_axis(sim, idx, axis=1)

    rows = []
    for i in range(q.shape[0]):
        for j in range(k):
            cat_idx = int(idx[i, j])
            rows.append({
                "query_rti": df_new.iloc[i]["rti_number"],
                "candidate_rti": catalog_meta.iloc[cat_idx]["rti_number"],
                "candidate_gti": catalog_meta.iloc[cat_idx]["gti_number"],
                "score": float(score[i, j]),
                "rank": j + 1,
            })
    return pd.DataFrame(rows)
