from __future__ import annotations

import csv
import multiprocessing as mp
import queue
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterator

from acfc.core.backends import build_backend
from acfc.core.manifest import MANIFEST, read_manifest, write_row
from acfc.core.phases import phase1_map, phase2_people, phase3_text
from acfc.core.pipeline import process_image

EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class RunConfig:
    input_dir: Path
    workers: int = 2
    batch_size: int = 5
    min_area_ratio: float = 0.4
    force: bool = False
    backend: str = "real"
    phase: int = 0                                   # 0 = una sola pasada, 1/2/3 = fases
    mode: str = "unico"
    phases: tuple = (1, 2, 3)
    sides: dict = field(default_factory=dict)        # fase 3: nombre -> lado del manifiesto

    @property
    def in_dir(self) -> Path:
        base = Path(self.input_dir)
        return {0: base, 1: base, 2: base / "fase1_mapa", 3: base / "fase2_personas"}[self.phase]

    @property
    def out_dir(self) -> Path:
        base = Path(self.input_dir)
        return {0: base / "outputs", 1: base / "fase1_mapa", 2: base / "fase2_personas", 3: base / "outputs"}[self.phase]


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def make_batches(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _process(cfg: RunConfig, path: Path, people, overlay):
    if cfg.phase == 1:
        return phase1_map(path, cfg.out_dir, min_keep=cfg.min_area_ratio, force=cfg.force)
    if cfg.phase == 2:
        return phase2_people(path, cfg.out_dir, people, cfg.force)
    if cfg.phase == 3:
        return phase3_text(path, cfg.out_dir, overlay, cfg.sides.get(path.name, "ninguno"), cfg.min_area_ratio, cfg.force)
    return process_image(path, cfg.out_dir, people, overlay, cfg.min_area_ratio, cfg.force)


def worker_main(worker_id: int, task_q, event_q, cfg: RunConfig, cancel_ev) -> None:
    try:
        people, overlay = build_backend(cfg.backend, cfg.phase)
    except Exception as exc:
        event_q.put({"type": "worker_error", "worker": worker_id, "error": f"{type(exc).__name__}: {exc}"})
        event_q.put({"type": "worker_done", "worker": worker_id})
        return
    n = 0
    while True:
        batch = task_q.get()
        if batch is None or cancel_ev.is_set():
            break
        n += 1
        event_q.put({"type": "batch_start", "worker": worker_id, "size": len(batch), "batch": n})
        for p in batch:
            if cancel_ev.is_set():
                break
            res = _process(cfg, Path(p), people, overlay)
            event_q.put({"type": "image_done", "worker": worker_id, **asdict(res)})
    event_q.put({"type": "worker_done", "worker": worker_id})


class _Logs:
    BASE = {
        "discarded": ("descartadas", ["archivo", "motivo", "score_gorra", "score_torso"]),
        "error": ("errores", ["archivo", "error"]),
        "review": ("revisar", ["archivo", "area_ratio", "motivo"]),
    }

    def __init__(self, out_dir: Path, phase: int = 0):
        self.out_dir, self.phase = out_dir, phase

    def filename(self, kind: str) -> str:
        suffix = self.BASE[kind][0]
        return f"log_{suffix}.csv" if self.phase == 0 else f"log_fase{self.phase}_{suffix}.csv"

    def all_files(self) -> list[Path]:
        return [self.out_dir / self.filename(k) for k in self.BASE]

    def write(self, ev: dict) -> None:
        kind = ev["status"]
        if kind not in self.BASE:
            return
        header = self.BASE[kind][1]
        path = self.out_dir / self.filename(kind)
        new = not path.exists()
        row = {
            "discarded": [ev["name"], ev["reason"], f'{ev["head"]:.3f}', f'{ev["torso"]:.3f}'],
            "error": [ev["name"], ev["reason"]],
            "review": [ev["name"], f'{ev["area_ratio"]:.3f}', ev["reason"]],
        }[kind]
        with path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(header)
            w.writerow(row)


class Runner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self._ctx = mp.get_context("spawn")
        self._cancel = self._ctx.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def events(self) -> Iterator[dict]:
        cfg = self.cfg
        if not cfg.in_dir.is_dir():
            raise FileNotFoundError(f"falta la carpeta de la fase anterior: {cfg.in_dir}")
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.force:  # al rehacer, los logs y el manifiesto de la corrida anterior ya no valen
            for f in _Logs(cfg.out_dir, cfg.phase).all_files():
                f.unlink(missing_ok=True)
            if cfg.phase in (1, 2):
                (cfg.out_dir / MANIFEST).unlink(missing_ok=True)
        files = list_images(cfg.in_dir)
        prev: dict = {}
        if cfg.phase == 3:
            cfg = replace(cfg, sides={n: sd for n, (sd, _c) in read_manifest(cfg.in_dir).items()})
        elif cfg.phase == 2:
            prev = read_manifest(cfg.in_dir)
        batches = make_batches([str(p) for p in files], cfg.batch_size)
        n_workers = min(cfg.workers, len(batches))
        yield {"type": "start", "fase": cfg.phase, "total": len(files), "workers": n_workers, "batches": len(batches)}

        counts: Counter = Counter()
        if n_workers:
            task_q, event_q = self._ctx.Queue(), self._ctx.Queue()
            for b in batches:
                task_q.put(b)
            for _ in range(n_workers):
                task_q.put(None)
            procs = [self._ctx.Process(target=worker_main, args=(i, task_q, event_q, cfg, self._cancel), daemon=True)
                     for i in range(n_workers)]
            for p in procs:
                p.start()
            logs = _Logs(cfg.out_dir, cfg.phase)
            finished: set[int] = set()
            dead_polls = [0] * n_workers
            while len(finished) < n_workers:
                try:
                    ev = event_q.get(timeout=0.5)
                except queue.Empty:
                    for i, p in enumerate(procs):
                        if i in finished:
                            continue
                        dead_polls[i] = dead_polls[i] + 1 if not p.is_alive() else 0
                        if dead_polls[i] >= 3:
                            finished.add(i)
                            yield {"type": "worker_error", "worker": i, "error": "el worker termino inesperadamente"}
                            yield {"type": "worker_done", "worker": i}
                    continue
                if ev["type"] == "image_done":
                    counts[ev["status"]] += 1
                    logs.write(ev)
                    if cfg.phase == 1 and ev["status"] in ("cropped", "copied", "review"):
                        write_row(cfg.out_dir, ev["name"], ev["side"], ev["cut"])
                    elif cfg.phase == 2 and ev["status"] == "copied":
                        side, cut = prev.get(ev["name"], ("ninguno", 0))
                        write_row(cfg.out_dir, ev["name"], side, cut)
                elif ev["type"] == "worker_done":
                    finished.add(ev["worker"])
                yield ev
            for p in procs:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
        yield {"type": "finished", "fase": cfg.phase, "counts": dict(counts), "cancelled": self._cancel.is_set()}
