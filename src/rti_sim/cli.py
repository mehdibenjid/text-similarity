from __future__ import annotations
import json
from pathlib import Path
import typer
from rti_sim.settings import load_settings
from rti_sim.logging import get_logger
from rti_sim.pipelines import stage0_ingest
from rti_sim.pipelines import stage1_preprocess  # ← this was missing

app = typer.Typer(add_completion=False)
log = get_logger()

@app.command()
def train(config: str = typer.Option("configs/base.yaml", "--config", "-c", help="Path to YAML config")):
    cfg = load_settings(config)
    meta0 = stage0_ingest.run(cfg)
    meta1 = stage1_preprocess.run(cfg)  # ← actually run Stage 1
    log.info("[train] Phase 1 complete.")
    typer.echo(json.dumps({**meta0, **meta1}, indent=2))

@app.command()
def infer(
    input: str = typer.Option(..., "--input", "-i", help="Path to new RTIs (parquet/csv)"),
    out: str = typer.Option(..., "--out", "-o", help="Where to write assignments"),
    config: str = typer.Option("configs/base.yaml", "--config", "-c", help="Path to YAML config"),
):
    _ = load_settings(config)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    log.info("[infer] Phase 0 stub: no model yet; writing empty result.")
    Path(out).write_text("[]\n", encoding="utf-8")
    typer.echo(out)

def main():
    app()

if __name__ == "__main__":
    main()
