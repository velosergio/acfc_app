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
    right_align_tol: float = 0.03    # los bordes derechos de las líneas deben coincidir (relativo a W)
    chain_extend: float = 1.5        # alturas de línea que se extienden más allá de la última línea vista
    min_ocr_conf: float = 0.30       # una línea es "fuerte" con esta confianza o más
    line_h_min: float = 0.012        # alto de línea de texto relativo al ancho de la imagen
    line_h_max: float = 0.07
    chain_gap: float = 4.0           # separación máxima entre líneas (en alturas de línea)
    min_lines: int = 2               # líneas fuertes mínimas en el bloque
    min_digits: int = 3              # el bloque GPS siempre trae fecha/coordenadas
    text_pad_left: float = 0.03      # margen extra a la izquierda del bloque (por si el OCR perdió la línea más larga)
    margin: float = 0.012            # margen extra alrededor de cada caja
    # minimapa (pin rojo de Google en el centro del mosaico)
    pin_area_min: float = 0.0004     # relativo a W²
    pin_area_max: float = 0.004
    tile_min_w: float = 0.12         # ancho del mosaico relativo a W
    tile_max_w: float = 0.45
    tile_aspect_min: float = 0.7     # alto/ancho del mosaico
    tile_aspect_max: float = 1.4
    tile_fill: float = 0.5           # fracción de píxeles "color mapa" alrededor del pin
    # brújula (aguja cian)
    needle_area_min: float = 0.0003  # relativo a W²
    compass_size: float = 0.21       # lado de la brújula relativo a W


def _alnum(text: str) -> int:
    return len(re.findall(r"\w", text))


def text_chain(results, width: int, height: int, y_offset: int, at_top: bool,
               p: OverlayParams) -> tuple[Box, int] | None:
    """Bloque de texto de la app GPS: líneas alineadas a la derecha, pegadas al borde superior o inferior.

    Devuelve (caja, nº de líneas fuertes) o None.
    """
    lines = []  # (x0, y0, x1, y1, fuerte, dígitos)
    for pts, text, conf in results:
        if conf < p.weak_conf or _alnum(text) < 2:
            continue
        px = [int(pt[0]) for pt in pts]
        py = [int(pt[1]) + y_offset for pt in pts]
        h = max(py) - min(py)
        if max(px) < p.right_align_min * width or not (p.line_h_min * width <= h <= p.line_h_max * width):
            continue
        lines.append((min(px), min(py), max(px), max(py), conf >= p.min_ocr_conf, sum(ch.isdigit() for ch in text)))
    if not lines:
        return None
    med_right = float(np.median([b[2] for b in lines]))
    lines = [b for b in lines if abs(b[2] - med_right) <= p.right_align_tol * width]
    if not lines:
        return None
    lines.sort(key=(lambda b: b[1]) if at_top else (lambda b: -b[3]))
    edge_dist = lines[0][1] if at_top else height - lines[0][3]
    if edge_dist > p.anchor_max * height:
        return None
    chain = [lines[0]]
    for b in lines[1:]:
        med_h = float(np.median([c[3] - c[1] for c in chain]))
        gap = (b[1] - max(c[3] for c in chain)) if at_top else (min(c[1] for c in chain) - b[3])
        if gap <= p.chain_gap * med_h:
            chain.append(b)
    strong = sum(1 for c in chain if c[4])
    if strong < p.min_lines or sum(c[5] for c in chain) < p.min_digits:
        return None
    x0 = max(0, min(c[0] for c in chain) - int(p.text_pad_left * width))
    ext = int(p.chain_extend * float(np.median([c[3] - c[1] for c in chain])))
    if at_top:
        y0, y1 = 0, min(height, max(c[3] for c in chain) + ext)
    else:
        y0, y1 = max(0, min(c[1] for c in chain) - ext), height
    return (x0, y0, width, y1), strong


def text_boxes_from_ocr(results, width: int, height: int, y_offset: int, strip_h: int,
                        at_top: bool, p: OverlayParams) -> list[Box]:
    found = text_chain(results, width, height, y_offset, at_top, p)
    return [found[0]] if found else []


def _map_like_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return ((hsv[..., 1] <= 90) & (hsv[..., 2] >= 185)).astype(np.uint8)


def _components(mask: np.ndarray):
    n, _labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        yield int(x), int(y), int(w), int(h), int(area), cents[i]


def find_minimap(rgb: np.ndarray, p: OverlayParams) -> list[Box]:
    H, W = rgb.shape[:2]
    region_w, region_h = int(p.tile_max_w * W / 2) + 2, min(H, int(0.35 * W))
    boxes: list[Box] = []
    for right in (False, True):
        for bottom in (False, True):
            view = rgb[:, ::-1] if right else rgb
            view = view[::-1] if bottom else view
            crop = np.ascontiguousarray(view[:region_h, :region_w])
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            red = (((hsv[..., 0] <= 8) | (hsv[..., 0] >= 172)) & (hsv[..., 1] >= 140) & (hsv[..., 2] >= 150)).astype(np.uint8)
            mapish = _map_like_mask(crop)
            for x, y, w, h, area, (cx, _cy) in _components(red):
                if not (p.pin_area_min * W * W <= area <= p.pin_area_max * W * W) or not (1.0 <= h / max(w, 1) <= 2.6):
                    continue
                tip = y if bottom else y + h  # en la vista volteada la punta del pin queda arriba
                tw, th = int(2 * cx), int(2 * tip)
                if not (p.tile_min_w * W <= tw <= p.tile_max_w * W) or not (p.tile_aspect_min <= th / tw <= p.tile_aspect_max):
                    continue
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
    size = int(p.compass_size * W)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    cyan = ((hsv[..., 0] >= 84) & (hsv[..., 0] <= 98) & (hsv[..., 1] >= 140) & (hsv[..., 2] >= 120)).astype(np.uint8)
    boxes: list[Box] = []
    for right in (False, True):
        for bottom in (False, True):
            x0, x1 = (W - size, W) if right else (0, size)
            y0, y1 = (H - size, H) if bottom else (0, size)
            sub = cyan[max(y0, 0):y1, x0:x1]
            for _x, _y, w, h, area, _c in _components(sub):
                if area >= p.needle_area_min * W * W and w <= size * 0.5 and h <= size * 0.8:
                    boxes.append((x0, max(y0, 0), x1, y1))
                    break
    return boxes


class OverlayDetector:
    def __init__(self, gpu: bool = True, params: OverlayParams = OverlayParams()):
        import easyocr
        import torch

        self.p = params
        self.reader = easyocr.Reader(["es", "en"], gpu=gpu and torch.cuda.is_available(), verbose=False)

    def detect(self, rgb: np.ndarray) -> list[Box]:
        H, W = rgb.shape[:2]
        strip = int(self.p.strip_ratio * H)
        candidates = []
        for y0, at_top in ((0, True), (H - strip, False)):
            results = self.reader.readtext(np.ascontiguousarray(rgb[y0:y0 + strip]), low_text=0.3,
                                           text_threshold=0.5, contrast_ths=0.3, adjust_contrast=0.7)
            found = text_chain(results, W, H, y0, at_top, self.p)
            if found:
                candidates.append(found)
        boxes: list[Box] = [max(candidates, key=lambda c: c[1])[0]] if candidates else []  # un solo bloque por foto
        boxes += find_minimap(rgb, self.p) + find_compass(rgb, self.p)
        return [pad_box(b, W, H, self.p.margin) for b in boxes]
