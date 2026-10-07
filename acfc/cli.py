from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from acfc.core.runner import RunConfig, Runner


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="acfc", description="Descarta imagenes con uniforme+gorra blancos y recorta mapa/texto.")
    ap.add_argument("ruta", type=Path, help="carpeta con las imagenes")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-area", type=float, default=0.4, help="area minima del recorte (0-1) antes de marcar para revisar")
    ap.add_argument("--force", action="store_true", help="rehacer imagenes que ya existen en outputs")
    ap.add_argument("--backend", choices=["real", "fake"], default="real", help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ruta.is_dir():
        print(f"Error: la carpeta no existe: {args.ruta}", file=sys.stderr)
        return 2
    cfg = RunConfig(args.ruta, workers=max(1, args.workers), min_area_ratio=args.min_area,
                    force=args.force, backend=args.backend)
    runner = Runner(cfg)
    columns = (TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn())
    worker_tasks: dict[int, int] = {}
    last: dict = {}
    try:
        with Progress(*columns) as prog:
            overall = prog.add_task("General", total=0)
            for ev in runner.events():
                t = ev["type"]
                if t == "start":
                    prog.update(overall, total=ev["total"])
                    for w in range(ev["workers"]):
                        worker_tasks[w] = prog.add_task(f"Worker {w}", total=0)
                elif t == "batch_start":
                    prog.reset(worker_tasks[ev["worker"]], total=ev["size"],
                               description=f"Worker {ev['worker']} lote {ev['batch']}")
                elif t == "image_done":
                    prog.advance(worker_tasks[ev["worker"]])
                    prog.advance(overall)
                elif t == "worker_error":
                    prog.console.print(f"[red]Worker {ev['worker']}: {ev['error']}[/red]")
                elif t == "finished":
                    last = ev
    except KeyboardInterrupt:
        runner.cancel()
        print("Cancelando...", file=sys.stderr)
        return 130
    counts = last.get("counts", {})
    print(f"Listo. guardadas={counts.get('saved', 0)} revisar={counts.get('review', 0)} "
          f"descartadas={counts.get('discarded', 0)} errores={counts.get('error', 0)} "
          f"omitidas={counts.get('skipped', 0)}  ->  {cfg.out_dir}")
    return 0
