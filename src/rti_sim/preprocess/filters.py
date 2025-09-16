from __future__ import annotations
import pandas as pd

def filter_ac_msn(df: pd.DataFrame, value: str, drop_after: bool = True) -> pd.DataFrame:
    """Keep only rows where AC_MSN == value (string compare after strip)."""
    if "AC_MSN" not in df.columns:
        return df  # nothing to do
    tmp = df.copy()
    tmp["__ac_msn_norm__"] = tmp["AC_MSN"].fillna("").astype(str).str.strip()
    out = tmp[tmp["__ac_msn_norm__"] == str(value)].drop(columns="__ac_msn_norm__")
    if drop_after and "AC_MSN" in out.columns:
        out = out.drop(columns="AC_MSN")
    return out
