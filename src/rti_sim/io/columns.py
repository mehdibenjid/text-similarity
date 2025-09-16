from __future__ import annotations
import pandas as pd
from typing import Iterable, Mapping

def apply_aliases(df: pd.DataFrame, aliases: Mapping[str, str] | None) -> pd.DataFrame:
    """Rename columns using a provided alias map (raw -> canonical)."""
    if not aliases:
        return df
    present = {raw: to for raw, to in aliases.items() if raw in df.columns}
    return df.rename(columns=present)

def project(df: pd.DataFrame, keep: Iterable[str]) -> pd.DataFrame:
    """Keep only requested canonical columns; add missing as empty strings."""
    keep = list(keep or [])
    present = [c for c in keep if c in df.columns]
    out = df[present].copy()
    missing = [c for c in keep if c not in out.columns]
    for c in missing:
        out[c] = ""  # fill missing canonical columns to keep schema stable
    return out[keep]
