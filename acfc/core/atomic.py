from __future__ import annotations

import os
import shutil
from pathlib import Path

from PIL import Image

_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}


def save_atomic(img: Image.Image, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.tmp")
    fmt = _FORMATS.get(dest.suffix.lower(), "PNG")
    kwargs = {"quality": 95} if fmt == "JPEG" else {}
    img.save(tmp, format=fmt, **kwargs)
    os.replace(tmp, dest)


def copy_atomic(src: Path, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)
