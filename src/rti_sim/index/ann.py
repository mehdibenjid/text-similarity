from __future__ import annotations
import numpy as np
from typing import Tuple

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors

def build_index(emb: np.ndarray, backend: str = "faiss", metric: str = "cosine",
                faiss_use_ivf: bool = False, nlist: int = 1024) -> object:
    """
    Retourne un index ANN entrainé. Si backend=faiss mais indispo, fallback sklearn.
    Embeddings doivent idéalement être normalisés pour cosine/IP.
    """
    backend = backend.lower()
    metric = metric.lower()

    if backend == "faiss" and _HAS_FAISS:
        d = emb.shape[1]
        if metric == "cosine":
            # Avec emb normalisés, IP == cosine
            index = faiss.IndexFlatIP(d)
        elif metric in ("ip", "inner_product"):
            index = faiss.IndexFlatIP(d)
        else:
            # L2 peu utile pour embeddings, mais bon
            index = faiss.IndexFlatL2(d)

        if faiss_use_ivf:
            quantizer = index
            nlist = max(1, int(nlist))
            ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT if metric in ("cosine","ip","inner_product") else faiss.METRIC_L2)
            ivf.train(emb)
            ivf.add(emb)
            return ivf
        else:
            index.add(emb)
            return index

    # fallback sklearn
    algo = "brute" if metric == "cosine" else "auto"
    nn = NearestNeighbors(n_neighbors=10, algorithm=algo, metric=metric)
    nn.fit(emb)
    return nn

def query_index(index: object, q: np.ndarray, k: int = 10, metric: str = "cosine") -> Tuple[np.ndarray, np.ndarray]:
    if _HAS_FAISS and hasattr(index, "search"):
        # FAISS
        D, I = index.search(q, k)
        return I, D
    else:
        # sklearn
        # kneighbors retourne distances; si metric=cosine, c'est 1 - sim
        dist, idx = index.kneighbors(q, n_neighbors=k, return_distance=True)
        if metric.lower() == "cosine":
            sim = 1.0 - dist
            return idx, sim
        return idx, -dist  # convention: plus grand = plus proche
