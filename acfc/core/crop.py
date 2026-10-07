from __future__ import annotations

Box = tuple[int, int, int, int]


def pad_box(box: Box, width: int, height: int, margin: float) -> Box:
    m = int(margin * max(width, height))
    x0, y0, x1, y1 = box
    return (max(0, x0 - m), max(0, y0 - m), min(width, x1 + m), min(height, y1 + m))


def _overlaps(r: Box, b: Box) -> bool:
    return r[0] < b[2] and b[0] < r[2] and r[1] < b[3] and b[1] < r[3]


def largest_free_rect(width: int, height: int, boxes: list[Box]) -> Box | None:
    """Rectángulo de mayor área que no intersecta ninguna caja.

    Los bordes óptimos siempre coinciden con bordes de imagen o de cajas, así que
    basta probar todas las combinaciones de esas coordenadas.
    """
    xs = sorted({0, width, *(min(max(b[0], 0), width) for b in boxes), *(min(max(b[2], 0), width) for b in boxes)})
    ys = sorted({0, height, *(min(max(b[1], 0), height) for b in boxes), *(min(max(b[3], 0), height) for b in boxes)})
    best: Box | None = None
    best_area = 0
    for i, x0 in enumerate(xs):
        for x1 in xs[i + 1:]:
            for j, y0 in enumerate(ys):
                for y1 in ys[j + 1:]:
                    area = (x1 - x0) * (y1 - y0)
                    if area <= best_area:
                        continue
                    r = (x0, y0, x1, y1)
                    if not any(_overlaps(r, b) for b in boxes):
                        best, best_area = r, area
    return best
