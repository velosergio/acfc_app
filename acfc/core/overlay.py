from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from acfc.core.crop import Box, pad_box


@dataclass(frozen=True)
class OverlayParams:
    # texto
    strip_ratio: float = 0.42        # franja superior/inferior donde se busca texto
    weak_conf: float = 0.02          # las líneas débiles solo cuentan si se unen a la cadena
    anchor_max: float = 0.15         # la primera línea vista debe estar a menos de esta fracción de H del borde
    right_align_min: float = 0.90    # el borde derecho de cada línea debe pasar de este x relativo
    right_align_tol: float = 0.07    # los bordes derechos de las anclas deben coincidir (relativo a W)
    h_gap: float = 3.0               # separación horizontal máxima entre trozos de una línea (en alturas de línea)
    chain_extend: float = 1.5        # alturas de línea que se extienden más allá de la última línea vista
    min_ocr_conf: float = 0.30       # una línea es "fuerte" con esta confianza o más
    line_h_min: float = 0.012        # alto de línea de texto relativo al ancho de la imagen
    line_h_max: float = 0.10
    chain_gap: float = 4.0           # separación máxima entre líneas (en alturas de línea)
    min_lines: int = 2               # líneas fuertes mínimas en el bloque
    min_digits: int = 3              # el bloque GPS siempre trae fecha/coordenadas
    text_pad_left: float = 0.03      # margen extra a la izquierda del bloque (por si el OCR perdió la línea más larga)
    margin: float = 0.012            # margen extra alrededor de cada caja
    # minimapa (pin rojo de Google en el centro del mosaico)
    pin_area_min: float = 0.00012    # relativo al lado corto²
    pin_area_max: float = 0.004
    tile_min_w: float = 0.12         # ancho del mosaico relativo a W
    tile_max_w: float = 0.45
    tile_aspect_min: float = 0.7     # alto/ancho del mosaico
    tile_aspect_max: float = 1.6
    tile_edge: float = 25.0          # salto mínimo de color para reconocer el borde real del mosaico
    tile_fill: float = 0.5           # fracción de píxeles "color mapa" alrededor del pin
    # brújula (aguja cian)
    needle_area_min: float = 0.0003  # relativo a W²
    compass_size: float = 0.21       # lado de la brújula relativo a W


def _alnum(text: str) -> int:
    return len(re.findall(r"\w", text))


def text_chain(results, width: int, height: int, y_offset: int, at_top: bool,
               p: OverlayParams, min_lines: int | None = None) -> tuple[Box, int] | None:
    """Bloque de texto de la app GPS: líneas alineadas a la derecha, pegadas al borde superior o inferior.

    Devuelve (caja, nº de líneas fuertes) o None.
    """
    frags = []  # (x0, y0, x1, y1, conf, texto)
    for pts, text, conf in results:
        if conf < p.weak_conf or _alnum(text) < 1:
            continue
        px = [int(pt[0]) for pt in pts]
        py = [int(pt[1]) + y_offset for pt in pts]
        if p.line_h_min * min(width, height) <= max(py) - min(py) <= p.line_h_max * min(width, height):
            frags.append((min(px), min(py), max(px), max(py), conf, text))
    # anclas: fragmentos pegados al margen derecho, con borde derecho coincidente
    anchors = [f for f in frags if _alnum(f[5]) >= 2 and f[2] >= p.right_align_min * width]
    if not anchors:
        return None
    med_right = float(np.median([f[2] for f in anchors]))
    anchors = [f for f in anchors if abs(f[2] - med_right) <= p.right_align_tol * width]
    if not anchors:
        return None
    anchors.sort(key=(lambda f: f[1]) if at_top else (lambda f: -f[3]))
    edge_dist = anchors[0][1] if at_top else height - anchors[0][3]
    if edge_dist > p.anchor_max * height:
        return None
    chain = [anchors[0]]
    for f in anchors[1:]:
        med_h = float(np.median([c[3] - c[1] for c in chain]))
        gap = (f[1] - max(c[3] for c in chain)) if at_top else (min(c[1] for c in chain) - f[3])
        if gap <= p.chain_gap * med_h:
            chain.append(f)
    med_h = float(np.median([c[3] - c[1] for c in chain]))
    # miembros: fragmentos a la altura de la cadena y pegados horizontalmente (el OCR parte las líneas en trozos)
    y_lo, y_hi = min(c[1] for c in chain) - 0.5 * med_h, max(c[3] for c in chain) + 0.5 * med_h
    members = list(chain)
    left = min(c[0] for c in chain)
    changed = True
    while changed:
        changed = False
        for f in frags:
            if f in members or not (y_lo <= (f[1] + f[3]) / 2 <= y_hi):
                continue
            if f[2] >= left - p.h_gap * med_h:
                members.append(f)
                left = min(left, f[0])
                changed = True
    strong = sum(1 for c in chain if c[4] >= p.min_ocr_conf)
    if strong < (p.min_lines if min_lines is None else min_lines) or sum(sum(ch.isdigit() for ch in m[5]) for m in members) < p.min_digits:
        return None
    x0 = max(0, left - int(p.text_pad_left * width))
    ext = int(p.chain_extend * med_h)
    if at_top:
        y0, y1 = 0, min(height, max(m[3] for m in members) + ext)
    else:
        y0, y1 = max(0, min(m[1] for m in members) - ext), height
    return (x0, y0, width, y1), strong


def text_boxes_from_ocr(results, width: int, height: int, y_offset: int, strip_h: int,
                        at_top: bool, p: OverlayParams, min_lines: int | None = None) -> list[Box]:
    found = text_chain(results, width, height, y_offset, at_top, p, min_lines)
    return [found[0]] if found else []


def text_boxes_from_detection(boxes, width: int, height: int, y_offset: int, at_top: bool,
                              p: OverlayParams, min_lines: int | None = None) -> list[Box]:
    """Igual que text_boxes_from_ocr pero sin leer el texto: solo cuentan la geometría (líneas alineadas
    a la derecha, pegadas al borde y apiladas). boxes = (x0, y0, x1, y1) locales a la franja."""
    results = [([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], "0000", 1.0) for x0, y0, x1, y1 in boxes]
    found = text_chain(results, width, height, y_offset, at_top, p, min_lines)
    return [found[0]] if found else []


def _map_like_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return ((hsv[..., 1] <= 110) & (hsv[..., 2] >= 150)).astype(np.uint8)


def _components(mask: np.ndarray):
    n, _labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        yield int(x), int(y), int(w), int(h), int(area), cents[i]


def _refine_height(view: np.ndarray, tw: int, th: int, p: OverlayParams) -> int:
    """El pin no siempre queda en el centro exacto: busca el borde real del mosaico cerca de la estimación."""
    lo, hi = int(0.75 * th), min(view.shape[0] - 1, int(1.25 * th))
    if hi - lo < 3:
        return th
    cols = view[lo:hi + 1, int(0.1 * tw):max(int(0.9 * tw), int(0.1 * tw) + 1)].astype(np.int32)
    prof = cols.mean(axis=1)
    jump = np.abs(np.diff(prof, axis=0)).sum(axis=1)
    k = int(np.argmax(jump))
    return lo + k + 1 if jump[k] >= p.tile_edge else th


def find_minimap(rgb: np.ndarray, p: OverlayParams) -> list[Box]:
    H, W = rgb.shape[:2]
    s = min(H, W)
    region_w, region_h = int(p.tile_max_w * s / 2) + 2, min(H, int(0.35 * s))
    boxes: list[Box] = []
    for right in (False, True):
        for bottom in (False, True):
            view = rgb[:, ::-1] if right else rgb
            view = view[::-1] if bottom else view
            crop = np.ascontiguousarray(view[:region_h, :region_w])
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            red = (((hsv[..., 0] <= 8) | (hsv[..., 0] >= 172)) & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 150)).astype(np.uint8)
            mapish = _map_like_mask(crop)
            for x, y, w, h, area, (cx, _cy) in _components(red):
                if not (p.pin_area_min * s * s <= area <= p.pin_area_max * s * s) or not (1.0 <= h / max(w, 1) <= 2.6):
                    continue
                tip = y if bottom else y + h  # en la vista volteada la punta del pin queda arriba
                tw, th = int(2 * cx), int(2 * tip)
                if not (p.tile_min_w * s <= tw <= p.tile_max_w * s) or not (p.tile_aspect_min <= th / tw <= p.tile_aspect_max):
                    continue
                th = _refine_height(view, tw, th, p)
                m = int(1.5 * h)
                win = mapish[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
                red_win = red[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
                other = win[red_win == 0]
                if other.size == 0 or float(other.mean()) < p.tile_fill:
                    continue
                x0, x1 = (W - tw, W) if right else (0, tw)
                y0, y1 = (H - th, H) if bottom else (0, th)
                boxes.append((x0, max(y0, 0), x1, min(y1, H)))
                break
    return boxes


def find_compass(rgb: np.ndarray, p: OverlayParams) -> list[Box]:
    H, W = rgb.shape[:2]
    s = min(H, W)
    size = int(p.compass_size * s)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    cyan = ((hsv[..., 0] >= 84) & (hsv[..., 0] <= 98) & (hsv[..., 1] >= 140) & (hsv[..., 2] >= 120)).astype(np.uint8)
    boxes: list[Box] = []
    for right in (False, True):
        for bottom in (False, True):
            x0, x1 = (W - size, W) if right else (0, size)
            y0, y1 = (H - size, H) if bottom else (0, size)
            sub = cyan[max(y0, 0):y1, x0:x1]
            for _x, _y, w, h, area, _c in _components(sub):
                if area >= p.needle_area_min * s * s and w <= size * 0.5 and h <= size * 0.8:
                    boxes.append((x0, max(y0, 0), x1, y1))
                    break
    return boxes


class OverlayDetector:
    def __init__(self, gpu: bool = True, params: OverlayParams = OverlayParams(), recognizer: bool = True):
        import easyocr
        import torch

        self.p = params
        # recognizer=False carga solo el detector de texto (la fase 3 no lee nada): menos RAM y VRAM
        self.reader = easyocr.Reader(["es", "en"], gpu=gpu and torch.cuda.is_available(), verbose=False,
                                     recognizer=recognizer)

    def text_boxes(self, rgb: np.ndarray, sides: tuple[str, ...]) -> list[Box]:
        """Bloque de texto GPS solo por detección (sin reconocer). Con el lado del corte conocido basta una línea."""
        H, W = rgb.shape[:2]
        strip = int(self.p.strip_ratio * H)
        candidates = []
        min_lines = 1 if len(sides) == 1 else None
        for y0, at_top, lado in ((0, True, "arriba"), (H - strip, False, "abajo")):
            if lado not in sides:
                continue
            horizontal, free = self.reader.detect(np.ascontiguousarray(rgb[y0:y0 + strip]),
                                                  text_threshold=0.5, low_text=0.3)
            boxes = [(x0, y_0, x1, y_1) for x0, x1, y_0, y_1 in horizontal[0]]
            for pts in free[0]:
                xs, ys = [q[0] for q in pts], [q[1] for q in pts]
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
            found = text_chain([([(x0, y0_), (x1, y0_), (x1, y1_), (x0, y1_)], "0000", 1.0) for x0, y0_, x1, y1_ in boxes],
                               W, H, y0, at_top, self.p, min_lines)
            if found:
                candidates.append(found)
        if not candidates:
            return []
        return [pad_box(max(candidates, key=lambda c: c[1])[0], W, H, self.p.margin)]

    def text_boxes_ocr(self, rgb: np.ndarray, sides: tuple[str, ...]) -> list[Box]:
        H, W = rgb.shape[:2]
        strip = int(self.p.strip_ratio * H)
        candidates = []
        # con el lado ya conocido (tras el corte del mapa) suele quedar una sola línea legible: basta con una
        min_lines = 1 if len(sides) == 1 else None
        for y0, at_top, lado in ((0, True, "arriba"), (H - strip, False, "abajo")):
            if lado not in sides:
                continue
            results = self.reader.readtext(np.ascontiguousarray(rgb[y0:y0 + strip]), low_text=0.3,
                                           text_threshold=0.5, contrast_ths=0.3, adjust_contrast=0.7, batch_size=8)
            found = text_chain(results, W, H, y0, at_top, self.p, min_lines)
            if found:
                candidates.append(found)
        if not candidates:
            return []
        return [pad_box(max(candidates, key=lambda c: c[1])[0], W, H, self.p.margin)]  # un solo bloque por foto

    def detect(self, rgb: np.ndarray) -> list[Box]:
        H, W = rgb.shape[:2]
        boxes = self.text_boxes_ocr(rgb, ("arriba", "abajo"))
        boxes += [pad_box(b, W, H, self.p.margin) for b in find_minimap(rgb, self.p) + find_compass(rgb, self.p)]
        return boxes
