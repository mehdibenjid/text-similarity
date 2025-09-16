from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import typer
from rti_sim.settings import load_settings
from rti_sim.logging import get_logger
from rti_sim.pipelines import stage0_ingest, stage1_preprocess
from rti_sim.pipelines import stage2_embed
from rti_sim.pipelines.infer_knn import load_index_and_catalog, knn_assign
from rti_sim.embedding.encoder import TextEncoder
from rti_sim.pipelines import stage3_cluster
from rti_sim.pipelines.assign_cluster import run_assign

app = typer.Typer(add_completion=False)
log = get_logger()

def _cfg_dict(cfg):
    return cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()

@app.command()
def train(config: str = typer.Option("configs/base.yaml", "--config", "-c", help="Path to YAML config")):
    cfg = load_settings(config)
    meta0 = stage0_ingest.run(cfg)
    log.info("[train] Phase 0 complete.")
    meta1 = stage1_preprocess.run(cfg)
    log.info("[train] Phase 1 complete.")
    meta2 = stage2_embed.run(cfg)
    log.info("[train] Phase 2 complete.")
    typer.echo(json.dumps({**meta0, **meta1, **meta2}, indent=2))

@app.command()
def infer(
    input: str = typer.Option(..., "--input", "-i", help="Path to new RTIs views-like parquet/csv"),
    out: str = typer.Option(..., "--out", "-o", help="Where to write assignments (parquet)"),
    config: str = typer.Option("configs/base.yaml", "--config", "-c", help="Path to YAML config"),
    k: int = typer.Option(5, "--k", help="Top-K neighbors"),
):
    """
    Attendu en entrée: un fichier avec au minimum les colonnes
    ['rti_number','gti_number','title_text','chap_titles_text','meta_text'].
    Si tu veux, tu peux juste réutiliser data/processed/views.parquet pour tester.
    """
    cfg = load_settings(config)
    cfgd = _cfg_dict(cfg)

    in_path = Path(input)
    if in_path.suffix.lower() == ".csv":
        df_new = pd.read_csv(in_path, dtype=str)
    else:
        df_new = pd.read_parquet(in_path)

    emb_cfg = cfgd.get("embeddings", {}) or {}
    encoder = TextEncoder(
        model_name=str(emb_cfg.get("model_name", "intfloat/multilingual-e5-base")),
        device=str(emb_cfg.get("device", "auto")),
        normalize=bool(emb_cfg.get("normalize", True)),
        batch_size=int(emb_cfg.get("batch_size", 64)),
    )

    ann_cfg = cfgd.get("ann", {}) or {}
    backend = ann_cfg.get("backend", "faiss")
    metric = ann_cfg.get("metric", "cosine")

    out_dir = Path(cfg.paths.out_dir)
    catalog_emb, catalog_meta, index_obj = load_index_and_catalog(out_dir, backend=backend)
    assign = knn_assign(
        df_new=df_new,
        catalog_emb=catalog_emb,
        catalog_meta=catalog_meta,
        index_obj=index_obj,
        encoder=encoder,
        k=k,
        metric=metric,
    )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assign.to_parquet(out_path, index=False)
    typer.echo(str(out_path))

@app.command()
def cluster(config: str = typer.Option("configs/base.yaml", "--config", "-c")):
    cfg = load_settings(config)
    meta3 = stage3_cluster.run(cfg)
    typer.echo(json.dumps(meta3, indent=2))

@app.command("assign-clusters")
def assign_clusters(
    input: str = typer.Option(..., "--input", "-i", help="views-like parquet/csv to assign"),
    out: str = typer.Option(..., "--out", "-o", help="Output parquet with assignments"),
    config: str = typer.Option("configs/base.yaml", "--config", "-c"),
):
    cfg = load_settings(config)
    cfgd = _cfg_dict(cfg)
    emb_cfg = cfgd.get("embeddings", {}) or {}
    cl_cfg = cfgd.get("cluster", {}) or {}
    assign_cfg = cl_cfg.get("assign", {}) or {}

    df = run_assign(
        new_views_path=Path(input),
        out_dir=Path(cfg.paths.out_dir),
        model_name=str(emb_cfg.get("model_name", "intfloat/multilingual-e5-base")),
        device=str(emb_cfg.get("device", "auto")),
        normalize=bool(emb_cfg.get("normalize", True)),
        batch_size=int(emb_cfg.get("batch_size", 64)),
        metric=str(assign_cfg.get("metric", "cosine")),
        outlier_threshold=assign_cfg.get("outlier_threshold", None),
    )
    out_path = Path(out); out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    typer.echo(str(out_path))


def main():
    app()

if __name__ == "__main__":
    main()
