from __future__ import annotations
import numpy as np
import pandas as pd

def make_clustered_vectors(ids: list[str], gti: list[str], dim: int = 32) -> np.ndarray:
    """
    Renvoie des embeddings simples où les RTI qui partagent le même GTI
    sont proches: on fabrique un centre par GTI + un petit bruit.
    """
    rng = np.random.default_rng(42)
    gti_set = sorted(set(gti))
    centers = {g: rng.normal(0, 1, size=(dim,)).astype(np.float32) for g in gti_set}

    X = []
    for r, g in zip(ids, gti):
        base = centers[g] + rng.normal(0, 0.05, size=(dim,)).astype(np.float32)
        # normalisation L2 (comme dans le pipeline)
        n = np.linalg.norm(base) + 1e-12
        X.append((base / n).astype(np.float32))
    return np.stack(X, axis=0)

class DummyEncoder:
    def __init__(self, df_source: pd.DataFrame, dim: int = 32):
        self.df = df_source.reset_index(drop=True).copy()
        self.dim = dim
        # précompute vecteurs pour les RTI du catalogue
        self._X = make_clustered_vectors(
            self.df["rti_number"].tolist(),
            self.df["gti_number"].tolist(),
            dim=self.dim,
        )
        self._text_to_row = {i: i for i in range(len(self.df))}

    def encode(self, texts):
        """
        Dans les tests, on appelle encode avec la même longueur et le même ordre
        que df_source. On renvoie donc X tel quel. Pour d'autres inputs (infer),
        on génère les vecteurs en regroupant par GTI issu de df_source via merge.
        """
        # Si même taille -> assume ordre identique
        if len(texts) == len(self.df):
            return self._X
        # Sinon, on fait un mapping par rti_number dans le texte si possible
        # Attendu: chaque ligne de texte contient "rti_number=<ID>" (facultatif)
        import re
        out = []
        for t in texts:
            m = re.search(r"rti_number=([A-Za-z0-9_-]+)", str(t))
            if m:
                rid = m.group(1)
                row = self.df.index[self.df["rti_number"] == rid]
                if len(row):
                    out.append(self._X[row[0]])
                    continue
            # default: centre moyen + petit bruit
            out.append(np.mean(self._X, axis=0) + 0.0)
        return np.stack(out, axis=0).astype(np.float32)
