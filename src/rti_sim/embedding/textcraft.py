from __future__ import annotations
import pandas as pd

def build_embedding_text(
    df: pd.DataFrame,
    w_title: float = 2.0,
    w_chaps: float = 1.0,
    w_meta: float = 0.5,
    e5_prefix: str = "passage: ",
) -> pd.Series:
    """
    Concatène des champs texte avec pondération simple en répétant le texte au niveau LIGNE.
    Pas de vectorisation hasardeuse: on reste explicite et sûr.
    """
    def rep_scalar(text: str, times: float) -> str:
        if not text:
            return ""
        n = max(1, int(round(times)))
        return (" " + text) * n

    def row_text(row) -> str:
        title = (row.get("title_text") or "").strip()
        chaps = (row.get("chap_titles_text") or "").strip()
        meta  = (row.get("meta_text") or "").strip()

        parts = []
        parts.append("title:" + rep_scalar(title, w_title))
        parts.append("chapters:" + rep_scalar(chaps, w_chaps))
        parts.append("meta:" + rep_scalar(meta, w_meta))
        txt = " \n".join(parts).strip()
        return e5_prefix + txt if e5_prefix else txt

    # apply ligne par ligne → Series[str]
    return df.apply(row_text, axis=1)
