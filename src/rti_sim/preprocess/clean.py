from __future__ import annotations
import re
import unicodedata
import pandas as pd

_WS = re.compile(r"\s+")

def normalize_text(s: str | None) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("\u00A0", " ")
    s = _WS.sub(" ", s).strip()
    return s

def clean_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(normalize_text)
    return out
