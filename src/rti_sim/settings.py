from __future__ import annotations
from pydantic import BaseModel
from pathlib import Path
import yaml

class Paths(BaseModel):
    rti_master: str
    rti_chapters: str
    out_dir: str = "artifacts/"
    interim_dir: str = "data/interim/"      # ← default
    processed_dir: str = "data/processed/"  # ← default

class RunCfg(BaseModel):
    run_id_prefix: str = "dev"

class Settings(BaseModel):
    paths: Paths
    run: RunCfg = RunCfg()
    io: dict = {}
    columns: dict = {}
    filters: dict = {}
    text: dict = {}

def load_settings(path: str | Path) -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Settings(**data)
