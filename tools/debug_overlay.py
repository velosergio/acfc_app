import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image, ImageOps

from acfc.core.backends import build_backend
from acfc.core.crop import largest_free_rect
from acfc.core.runner import list_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    people, overlay = build_backend("real")
    files = list_images(a.folder)
    random.Random(a.seed).shuffle(files)
    for p in files[: a.n]:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        rgb = np.asarray(img)
        h, w = rgb.shape[:2]
        dec = people.check(rgb)
        boxes = overlay.detect(rgb)
        rect = largest_free_rect(w, h, boxes) if boxes else (0, 0, w, h)
        vis = np.ascontiguousarray(rgb[..., ::-1]).copy()
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 4)
        if rect:
            cv2.rectangle(vis, (rect[0], rect[1]), (rect[2] - 1, rect[3] - 1), (0, 255, 0), 4)
        cv2.putText(vis, f"descartar={dec.discard} gorra={dec.head:.2f} torso={dec.torso:.2f}",
                    (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, w / 900, (255, 0, 255), 3)
        cv2.imwrite(str(a.out / p.name), vis)
        print(p.name, "descartar" if dec.discard else "ok", f"{dec.head:.2f}/{dec.torso:.2f}", "cajas:", len(boxes))


if __name__ == "__main__":
    main()
