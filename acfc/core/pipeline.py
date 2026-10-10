from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from acfc.core.atomic import save_atomic
from acfc.core.crop import largest_free_rect


@dataclass
class ImageResult:
    name: str
    status: str  # saved | discarded | review | skipped | error
    reason: str = ""
    head: float = 0.0
    torso: float = 0.0
    area_ratio: float = 1.0
    side: str = ""
    cut: int = 0


def process_image(path: Path, out_dir: Path, people, overlay,
                  min_area_ratio: float = 0.4, force: bool = False) -> ImageResult:
    dest = out_dir / path.name
    try:
        if dest.exists() and not force:
            return ImageResult(path.name, "skipped", "ya existe")
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw).convert("RGB")
        rgb = np.asarray(img)
        decision = people.check(rgb)
        if decision.discard:
            return ImageResult(path.name, "discarded", "uniforme y gorra blancos",
                               decision.head, decision.torso)
        w, h = img.size
        boxes = overlay.detect(rgb)
        rect = largest_free_rect(w, h, boxes) if boxes else (0, 0, w, h)
        if rect is None:
            save_atomic(img, dest)
            return ImageResult(path.name, "review", "sin rectangulo libre; se guardo el original", area_ratio=1.0)
        ratio = (rect[2] - rect[0]) * (rect[3] - rect[1]) / (w * h)
        save_atomic(img.crop(rect), dest)
        if ratio < min_area_ratio:
            return ImageResult(path.name, "review", f"recorte pequeno ({ratio:.0%})", area_ratio=ratio)
        return ImageResult(path.name, "saved", area_ratio=ratio)
    except Exception as exc:  # una imagen mala nunca detiene la corrida
        return ImageResult(path.name, "error", f"{type(exc).__name__}: {exc}")
