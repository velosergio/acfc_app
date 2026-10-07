from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from acfc.core.crop import largest_free_rect
from acfc.core.overlay import OverlayParams, find_minimap
from acfc.core.pipeline import ImageResult, _save_atomic


def copy_atomic(src: Path, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def phase1_map(path: Path, out_dir: Path, params: OverlayParams = OverlayParams(),
               min_keep: float = 0.4, force: bool = False) -> ImageResult:
    dest = out_dir / path.name
    try:
        if dest.exists() and not force:
            return ImageResult(path.name, "skipped", "ya existe")
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw).convert("RGB")
        w, h = img.size
        maps = find_minimap(np.asarray(img), params)
        if not maps:
            copy_atomic(path, dest)
            return ImageResult(path.name, "copied", "sin minimapa", side="ninguno", cut=0)
        x0, y0, x1, y1 = max(maps, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        margin = int(params.margin * max(w, h))
        if y0 == 0:                                   # mapa arriba: se retira lo de arriba
            side, cut = "arriba", min(h, y1 + margin)
            rect = (0, cut, w, h)
        else:                                         # mapa abajo: se retira lo de abajo
            side, cut = "abajo", max(0, y0 - margin)
            rect = (0, 0, w, cut)
        keep = (rect[3] - rect[1]) / h
        _save_atomic(img.crop(rect), dest)
        status = "review" if keep < min_keep else "cropped"
        reason = f"corte deja {keep:.0%}" if status == "review" else ""
        return ImageResult(path.name, status, reason, area_ratio=keep, side=side, cut=cut)
    except Exception as exc:  # una imagen mala nunca detiene la corrida
        return ImageResult(path.name, "error", f"{type(exc).__name__}: {exc}")


def phase2_people(path: Path, out_dir: Path, people, force: bool = False) -> ImageResult:
    dest = out_dir / path.name
    try:
        if dest.exists() and not force:
            return ImageResult(path.name, "skipped", "ya existe")
        with Image.open(path) as raw:
            rgb = np.asarray(ImageOps.exif_transpose(raw).convert("RGB"))
        d = people.check(rgb)
        if d.discard:
            return ImageResult(path.name, "discarded", "uniforme y gorra blancos", d.head, d.torso)
        copy_atomic(path, dest)
        return ImageResult(path.name, "copied", "", d.head, d.torso)
    except Exception as exc:
        return ImageResult(path.name, "error", f"{type(exc).__name__}: {exc}")


def phase3_text(path: Path, out_dir: Path, overlay, side: str = "ninguno",
                min_area_ratio: float = 0.4, force: bool = False) -> ImageResult:
    dest = out_dir / path.name
    try:
        if dest.exists() and not force:
            return ImageResult(path.name, "skipped", "ya existe")
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw).convert("RGB")
        w, h = img.size
        sides = (side,) if side in ("arriba", "abajo") else ("arriba", "abajo")
        boxes = overlay.text_boxes(np.asarray(img), sides)
        if not boxes:
            copy_atomic(path, dest)
            return ImageResult(path.name, "saved", "sin texto", area_ratio=1.0)
        rect = largest_free_rect(w, h, boxes)
        if rect is None:
            _save_atomic(img, dest)
            return ImageResult(path.name, "review", "sin rectangulo libre; se guardo el original", area_ratio=1.0)
        ratio = (rect[2] - rect[0]) * (rect[3] - rect[1]) / (w * h)
        _save_atomic(img.crop(rect), dest)
        if ratio < min_area_ratio:
            return ImageResult(path.name, "review", f"recorte pequeno ({ratio:.0%})", area_ratio=ratio)
        return ImageResult(path.name, "saved", "", area_ratio=ratio)
    except Exception as exc:
        return ImageResult(path.name, "error", f"{type(exc).__name__}: {exc}")
