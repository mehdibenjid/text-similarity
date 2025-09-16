from __future__ import annotations
from pathlib import Path
import pandas as pd
from .contracts import RtiMasterSchema, RtiChaptersSchema
from .columns import apply_aliases, project

def _read_csv(path: Path, keep_canonical: list[str], aliases: dict | None, csv_opts: dict) -> pd.DataFrame:
    """
    Lire uniquement les colonnes brutes nécessaires :
    - un en-tête brut est gardé s'il mappe (via alias) vers un canonique que l'on garde
      ou s'il est déjà égal au nom canonique.
    """
    keep_set = set(keep_canonical or [])
    aliases = aliases or {}

    def usecols(name: str) -> bool:
        if name in aliases and aliases[name] in keep_set:
            return True
        if name in keep_set:
            return True
        return False

    uc = usecols if keep_set else None
    return pd.read_csv(path, dtype=str, usecols=uc, **(csv_opts or {}))

def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def _read_any(path: str | Path, keep_canonical: list[str], aliases: dict | None, csv_opts: dict) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return _read_csv(p, keep_canonical, aliases, csv_opts)
    return _read_parquet(p)

def load_master(path: str | Path, csv_opts: dict, keep: list[str], aliases_master: dict | None) -> pd.DataFrame:
    df = _read_any(path, keep, aliases_master, csv_opts)
    df = apply_aliases(df, aliases_master)   # brut -> canonique
    df = project(df, keep)                   # ne garder que les canoniques
    # robustesse: workload en int si présent
    if "workload" in df.columns:
        df["workload"] = pd.to_numeric(df["workload"], errors="coerce").astype("Int64")
    df = RtiMasterSchema.validate(df, lazy=True)
    # remettre en int natif si pas de NA
    if "workload" in df.columns and df["workload"].isna().sum() == 0:
        df["workload"] = df["workload"].astype(int)
    return df

def load_chapters(path: str | Path, csv_opts: dict, keep: list[str], aliases_chapters: dict | None) -> pd.DataFrame:
    df = _read_any(path, keep, aliases_chapters, csv_opts)
    df = apply_aliases(df, aliases_chapters)
    df = project(df, keep)
    # chapter_number int
    if "chapter_number" in df.columns:
        df["chapter_number"] = pd.to_numeric(df["chapter_number"], errors="coerce").astype("Int64")
    df = RtiChaptersSchema.validate(df, lazy=True)
    if "chapter_number" in df.columns and df["chapter_number"].isna().sum() == 0:
        df["chapter_number"] = df["chapter_number"].astype(int)
    return df
