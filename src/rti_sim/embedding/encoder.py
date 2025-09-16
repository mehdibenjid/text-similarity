from __future__ import annotations
from typing import Iterable, List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def _resolve_device(dev: str) -> str:
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev

class TextEncoder:
    def __init__(self, model_name: str, device: str = "auto", normalize: bool = True, batch_size: int = 64):
        self.device = _resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.normalize = bool(normalize)
        self.batch_size = int(batch_size)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,  # cosine-friendly
            show_progress_bar=True,
        )
        return emb.astype(np.float32, copy=False)
