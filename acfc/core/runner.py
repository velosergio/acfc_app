from __future__ import annotations

import csv
import multiprocessing as mp
import queue
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from acfc.core.backends import build_backend
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

    @property
    def out_dir(self) -> Path:
        return Path(self.input_dir) / "outputs"


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def make_batches(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def worker_main(worker_id: int, task_q, event_q, cfg: RunConfig, cancel_ev) -> None:
    try:
        people, overlay = build_backend(cfg.backend)
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
            res = process_image(Path(p), cfg.out_dir, people, overlay, cfg.min_area_ratio, cfg.force)
            event_q.put({"type": "image_done", "worker": worker_id, **asdict(res)})
    event_q.put({"type": "worker_done", "worker": worker_id})


class _Logs:
    FILES = {
        "discarded": ("log_descartadas.csv", ["archivo", "motivo", "score_gorra", "score_torso"]),
        "error": ("log_errores.csv", ["archivo", "error"]),
        "review": ("log_revisar.csv", ["archivo", "area_ratio", "motivo"]),
    }

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def write(self, ev: dict) -> None:
        spec = self.FILES.get(ev["status"])
        if not spec:
            return
        name, header = spec
        path = self.out_dir / name
        new = not path.exists()
        row = {
            "discarded": [ev["name"], ev["reason"], f'{ev["head"]:.3f}', f'{ev["torso"]:.3f}'],
            "error": [ev["name"], ev["reason"]],
            "review": [ev["name"], f'{ev["area_ratio"]:.3f}', ev["reason"]],
        }[ev["status"]]
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
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.force:  # al rehacer todo, los logs de la corrida anterior ya no valen
            for name, _header in _Logs.FILES.values():
                (cfg.out_dir / name).unlink(missing_ok=True)
        files = list_images(cfg.input_dir)
        batches = make_batches([str(p) for p in files], cfg.batch_size)
        n_workers = min(cfg.workers, len(batches))
        yield {"type": "start", "total": len(files), "workers": n_workers, "batches": len(batches)}

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
            logs = _Logs(cfg.out_dir)
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
                elif ev["type"] == "worker_done":
                    finished.add(ev["worker"])
                yield ev
            for p in procs:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
        yield {"type": "finished", "counts": dict(counts), "cancelled": self._cancel.is_set()}
