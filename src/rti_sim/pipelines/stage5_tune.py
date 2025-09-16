# src/rti_sim/pipelines/stage5_tune.py
from __future__ import annotations
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd

from rti_sim.logging import get_logger
from rti_sim.settings import Settings
from rti_sim.pipelines import stage2_embed, stage3_cluster
from rti_sim.pipelines.stage4_report import run as report_run

log = get_logger()


def _ensure_dict_section(cfg: Settings, attr: str) -> dict:
    """
    Garantit que cfg.<attr> est un dict modifiable.
    Si c'est un modèle Pydantic, on le convertit en dict.
    Si c'est None, on crée un dict vide.
    Retourne la section (dict) et réécrit cfg.<attr>.
    """
    section = getattr(cfg, attr, None)
    if section is None:
        setattr(cfg, attr, {})
        return getattr(cfg, attr)
    if isinstance(section, dict):
        return section
    # Pydantic ou autre objet
    try:
        as_dict = section.model_dump()  # pydantic v2
    except Exception:
        try:
            as_dict = section.dict()  # pydantic v1
        except Exception:
            as_dict = dict(section)
    setattr(cfg, attr, as_dict)
    return getattr(cfg, attr)


def _size_balance_from_assign(assign: pd.DataFrame) -> float:
    """
    Mesure simple d'équilibre des tailles de clusters.
    0 = très déséquilibré, 1 = plus équilibré.
    Utilise 1 - sum(p_i^2) (indice de Gini normalisé).
    """
    counts = assign["cluster_id"].value_counts().values.astype(float)
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    gini = 1.0 - np.sum(p ** 2)
    return float(gini)


def _score_row(purity: float, silhouette: float, size_balance: float) -> float:
    """
    Score agrégé (à ajuster selon métier).
    Silhouette est dans [-1, 1], on le remet dans [0, 1].
    """
    sil_norm = (silhouette + 1.0) / 2.0
    return 0.5 * purity + 0.3 * sil_norm + 0.2 * size_balance


def _get_cfg_dict(cfg: Settings) -> dict:
    # Utilitaire pour lire les valeurs par défaut du YAML
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()
    if hasattr(cfg, "dict"):
        return cfg.dict()
    return dict(cfg)


def run(cfg: Settings) -> dict:
    """
    Tuning de k et des pondérations texte (title/chapters/meta).
    Pour chaque combo: recalcul embeddings (stage2), clustering (stage3), reporting (stage4),
    puis calcule un score agrégé et écrit les résultats.

    Sorties:
      - artifacts/tuning_results.parquet
      - artifacts/tuning_best.json
    """
    cfgd = _get_cfg_dict(cfg)
    out_dir = Path(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grille de recherche (ou valeurs par défaut)
    tuning = cfgd.get("tuning", {}) or {}
    k_values = list(tuning.get("k_values", [4, 6, 8, 10, 12]))
    w_title_vals = list(tuning.get("w_title", [2.0]))
    w_chaps_vals = list(tuning.get("w_chaps", [1.0, 1.5]))
    w_meta_vals = list(tuning.get("w_meta", [0.5, 0.0]))

    # Sauvegarde des poids d'origine pour restauration en fin de run
    emb_section = _ensure_dict_section(cfg, "embeddings")
    base_weights = (emb_section.get("weights") or {}).copy()

    records = []
    combos = list(itertools.product(k_values, w_title_vals, w_chaps_vals, w_meta_vals))
    if not combos:
        raise ValueError("[tune] Empty search grid; check configs.tuning")

    for k, wt, wc, wm in combos:
        # 1) Met à jour les poids dans cfg.embeddings.weights
        emb_section = _ensure_dict_section(cfg, "embeddings")
        emb_section.setdefault("weights", {})
        emb_section["weights"]["title_text"] = float(wt)
        emb_section["weights"]["chap_titles_text"] = float(wc)
        emb_section["weights"]["meta_text"] = float(wm)

        # 2) Recalcule les embeddings (Stage 2)
        meta2 = stage2_embed.run(cfg)

        # 3) Set k dans cfg.cluster et lance Stage 3 (clustering)
        cl_section = _ensure_dict_section(cfg, "cluster")
        cl_section["k"] = int(k)
        meta3 = stage3_cluster.run(cfg)

        # 4) Lance Stage 4 (report) pour récupérer pureté moyenne etc.
        meta4 = report_run(cfg)

        # 5) Charge les assignations pour calculer un indicateur d'équilibre
        assign_path = out_dir / "cluster_assignments.parquet"
        assign = pd.read_parquet(assign_path) if assign_path.exists() else pd.DataFrame(columns=["cluster_id"])
        size_balance = _size_balance_from_assign(assign) if not assign.empty else 0.0

        purity = float(meta4.get("mean_purity", 0.0))
        silhouette = float(meta3.get("silhouette", -1.0))
        score = _score_row(purity, silhouette, size_balance)

        row = {
            "k": int(k),
            "w_title": float(wt),
            "w_chaps": float(wc),
            "w_meta": float(wm),
            "purity": purity,
            "silhouette": silhouette,
            "size_balance": size_balance,
            "score": float(score),
        }
        records.append(row)

        log.info(
            f"[tune] k={k} wt={wt} wc={wc} wm={wm} "
            f"→ score={score:.3f} (purity={purity:.3f}, sil={silhouette:.3f}, bal={size_balance:.3f})"
        )

    # Restaure les poids d'origine pour ne pas surprendre la suite de la pipeline
    emb_section = _ensure_dict_section(cfg, "embeddings")
    if base_weights:
        emb_section["weights"] = base_weights

    # Sauvegardes
    results = pd.DataFrame.from_records(records)
    results_path = out_dir / "tuning_results.parquet"
    results.to_parquet(results_path, index=False)

    best = results.sort_values("score", ascending=False).iloc[0].to_dict()
    best_path = out_dir / "tuning_best.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    return {
        "tuning_results_path": str(results_path),
        "tuning_best_path": str(best_path),
        **{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in best.items()},
    }
