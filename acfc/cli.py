from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from acfc.core.runner import RunConfig, make_runner

_ETIQUETAS = {"saved": "guardadas", "review": "revisar", "discarded": "descartadas", "error": "errores",
              "skipped": "omitidas", "cropped": "recortadas", "copied": "copiadas"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="acfc", description="Descarta imagenes con uniforme+gorra blancos y recorta mapa/texto.")
    ap.add_argument("ruta", type=Path, help="carpeta con las imagenes")
    ap.add_argument("--modo", choices=["fases", "unico"], default="fases",
                    help="fases: mapa -> personas -> texto (una carpeta por fase); unico: una sola pasada")
    ap.add_argument("--fase", choices=["1", "2", "3", "4", "todas"], default="todas", help="fase a ejecutar (modo fases)")
    ap.add_argument("--sin-personas", action="store_true",
                    help="saltar la fase 2 (personas): mapa -> texto; la fase 3 lee de fase1_mapa")
    ap.add_argument("--min-lado", type=int, default=500,
                    help="fase 4: se eliminan las imagenes con ancho o alto menor a estos px (por defecto 500)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-area", type=float, default=0.4, help="area minima del recorte (0-1) antes de marcar para revisar")
    ap.add_argument("--force", action="store_true", help="rehacer imagenes que ya existen en la carpeta de salida")
    ap.add_argument("--backend", choices=["real", "fake"], default="real", help=argparse.SUPPRESS)
    return ap


def _resumen(counts: dict) -> str:
    return " ".join(f"{_ETIQUETAS.get(k, k)}={v}" for k, v in sorted(counts.items())) or "sin imagenes"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ruta.is_dir():
        print(f"Error: la carpeta no existe: {args.ruta}", file=sys.stderr)
        return 2
    todas = (1, 3, 4) if args.sin_personas else (1, 2, 3, 4)
    phases = todas if args.fase == "todas" else (int(args.fase),)
    cfg = RunConfig(args.ruta, workers=max(1, args.workers), min_area_ratio=args.min_area,
                    force=args.force, backend=args.backend, mode=args.modo, phases=phases, skip_people=args.sin_personas, min_side=args.min_lado)
    runner = make_runner(cfg)
    columns = (TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn())
    worker_tasks: dict[int, int] = {}
    last: dict = {}
    try:
        with Progress(*columns) as prog:
            overall = prog.add_task("General", total=0)
            for ev in runner.events():
                t = ev["type"]
                if t == "start":
                    for tid in worker_tasks.values():
                        prog.remove_task(tid)
                    worker_tasks.clear()
                    fase = f" fase {ev['fase']}" if ev.get("fase") else ""
                    prog.reset(overall, total=ev["total"], description=f"General{fase}")
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
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        runner.cancel()
        print("Cancelando...", file=sys.stderr)
        return 130
    if last.get("phases"):
        for n, counts in sorted(last["phases"].items()):
            print(f"Fase {n}: {_resumen(counts)}  ->  {replace(cfg, phase=n).out_dir}")
    else:
        print(f"Listo. {_resumen(last.get('counts', {}))}  ->  {cfg.out_dir}")
    return 0
