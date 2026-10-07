import json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageOps

from acfc.core.overlay import OverlayParams, find_minimap
from acfc.core.runner import list_images


def main() -> None:
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows, c = {}, Counter()
    for f in list_images(folder):
        img = np.asarray(ImageOps.exif_transpose(Image.open(f)).convert("RGB"))
        maps = find_minimap(img, OverlayParams())
        rows[f.name] = {"W": img.shape[1], "H": img.shape[0], "maps": maps}
        c["sin minimapa" if not maps else ("arriba" if maps[0][1] == 0 else "abajo")] += 1
    out.write_text(json.dumps(rows))
    print(dict(c), "total", len(rows))
    for n in ("8028a0bab8273c91.jpeg", "cc6ae1bd9882ca07.jpeg"):
        print(n, "detectado" if rows.get(n, {}).get("maps") else "SIN DETECTAR")


if __name__ == "__main__":
    main()
