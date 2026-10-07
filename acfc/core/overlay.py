from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from acfc.core.crop import Box, pad_box


@dataclass(frozen=True)
class OverlayParams:
    strip_ratio: float = 0.30        # franja superior/inferior donde se busca texto
    right_align_min: float = 0.80    # el borde derecho de cada línea debe pasar de este x relativo
    min_ocr_conf: float = 0.15
    margin: float = 0.012            # margen extra alrededor de cada caja
    tile_corner_w: float = 0.45      # zona de esquina donde se busca el minimapa
    tile_corner_h: float = 0.40
    tile_min_w: float = 0.10
    tile_min_h: float = 0.05
    tile_fill: float = 0.90          # fracción mínima de píxeles "color mapa" dentro del rectángulo
    tile_edge: float = 25.0          # contraste mínimo en los bordes internos del rectángulo
    compass_corner: float = 0.30
    compass_ring_dark: float = 0.60  # fracción mínima del anillo oscuro y poco saturado


def text_boxes_from_ocr(results, width: int, y_offset: int, p: OverlayParams) -> list[Box]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for pts, _text, conf in results:
        if conf < p.min_ocr_conf:
            continue
        px = [int(pt[0]) for pt in pts]
        py = [int(pt[1]) for pt in pts]
        if max(px) < p.right_align_min * width:
            continue
        xs0.append(min(px)); ys0.append(min(py) + y_offset)
        xs1.append(max(px)); ys1.append(max(py) + y_offset)
    if not xs0:
        return []
    return [(min(xs0), min(ys0), max(xs1), max(ys1))]


def _map_like_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return ((hsv[..., 1] <= 90) & (hsv[..., 2] >= 185)).astype(np.uint8)


def _edge_contrast(a: np.ndarray, b: np.ndarray) -> float:
    """a y b: franjas (n, k, 3) a ambos lados de un borde; contraste mediano de color medio."""
    return float(np.median(np.abs(a.astype(np.int32).mean(axis=1) - b.astype(np.int32).mean(axis=1)).sum(axis=-1)))


def _tile_in_crop(crop: np.ndarray, full_w: int, full_h: int, p: OverlayParams) -> tuple[int, int] | None:
    """Busca un rectángulo anclado en la esquina (0,0) del recorte que parezca un minimapa."""
    ch, cw = crop.shape[:2]
    step = max(2, full_w // 200)
    min_w, min_h = int(p.tile_min_w * full_w), int(p.tile_min_h * full_h)
    ws = np.arange(min_w, cw - 8, step)
    hs = np.arange(min_h, ch - 8, step)
    if len(ws) == 0 or len(hs) == 0:
        return None
    ii = cv2.integral(_map_like_mask(crop)).astype(np.float64)
    area = hs[:, None] * ws[None, :]
    fill = ii[np.ix_(hs, ws)] / area
    cand = np.argwhere(fill >= p.tile_fill)
    if len(cand) == 0:
        return None
    order = np.argsort(-area[cand[:, 0], cand[:, 1]])
    for k in order[:400]:
        hi, wi = cand[k]
        w, h = int(ws[wi]), int(hs[hi])
        right = _edge_contrast(crop[:h, w - 4:w].transpose(1, 0, 2), crop[:h, w:w + 4].transpose(1, 0, 2))
        bottom = _edge_contrast(crop[h - 4:h, :w], crop[h:h + 4, :w])
        if right >= p.tile_edge and bottom >= p.tile_edge:
            return w, h
    return None


def find_minimap(rgb: np.ndarray, p: OverlayParams) -> list[Box]:
    H, W = rgb.shape[:2]
    cw, chh = int(p.tile_corner_w * W), int(p.tile_corner_h * H)
    boxes: list[Box] = []
    for right in (False, True):
        for bottom in (False, True):
            view = rgb[:, ::-1] if right else rgb
            view = view[::-1] if bottom else view
            found = _tile_in_crop(np.ascontiguousarray(view[:chh, :cw]), W, H, p)
            if not found:
                continue
            w, h = found
            x0, x1 = (W - w, W) if right else (0, w)
            y0, y1 = (H - h, H) if bottom else (0, h)
            boxes.append((x0, y0, x1, y1))
    return boxes


def find_compass(rgb: np.ndarray, p: OverlayParams) -> list[Box]:
    H, W = rgb.shape[:2]
    c = int(p.compass_corner * W)
    boxes: list[Box] = []
    for ox in (0, W - c):
        for oy in (0, H - c):
            crop = rgb[oy:oy + c, ox:ox + c]
            gray = cv2.medianBlur(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), 5)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=c, param1=120, param2=40,
                                       minRadius=int(0.04 * W), maxRadius=int(0.14 * W))
            if circles is None:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            for cx, cy, r in circles[0]:
                ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
                px = (cx + 0.93 * r * np.cos(ang)).astype(int)
                py = (cy + 0.93 * r * np.sin(ang)).astype(int)
                ok = (px >= 0) & (px < c) & (py >= 0) & (py < c)
                if ok.sum() < 20:
                    continue
                s, v = hsv[py[ok], px[ok], 1], hsv[py[ok], px[ok], 2]
                if float(((v < 120) & (s < 80)).mean()) >= p.compass_ring_dark:
                    box = (max(0, int(cx - r)) + ox, max(0, int(cy - r)) + oy,
                           min(c, int(cx + r)) + ox, min(c, int(cy + r)) + oy)
                    boxes.append(box)
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
        boxes: list[Box] = []
        for y0, y1 in ((0, strip), (H - strip, H)):
            results = self.reader.readtext(np.ascontiguousarray(rgb[y0:y1]))
            boxes += text_boxes_from_ocr(results, W, y0, self.p)
        boxes += find_minimap(rgb, self.p) + find_compass(rgb, self.p)
        return [pad_box(b, W, H, self.p.margin) for b in boxes]
