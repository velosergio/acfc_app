from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from acfc.core.crop import Box


@dataclass(frozen=True)
class PersonThresholds:
    head: float = 0.20
    torso: float = 0.22
    min_height_ratio: float = 0.08


@dataclass(frozen=True)
class PeopleDecision:
    discard: bool
    head: float = 0.0
    torso: float = 0.0


def white_fraction(rgb: np.ndarray, s_max: int = 50, v_min: int = 175) -> float:
    if rgb.size == 0:
        return 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[..., 1] <= s_max) & (hsv[..., 2] >= v_min)
    return float(mask.mean())


def _zone(rgb: np.ndarray, box: Box, fx0: float, fy0: float, fx1: float, fy1: float) -> np.ndarray:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    zx0, zx1 = int(x0 + fx0 * w), int(x0 + fx1 * w)
    zy0, zy1 = int(y0 + fy0 * h), int(y0 + fy1 * h)
    return rgb[max(zy0, 0):zy1, max(zx0, 0):zx1]


def _background_white(rgb: np.ndarray, box: Box, fy0: float, fy1: float) -> float:
    """Fracción de blanco en bandas a ambos lados de la persona, a la altura de la zona medida."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    ya, yb = max(0, int(y0 + fy0 * h)), int(y0 + fy1 * h)
    pad = int(0.4 * w)
    bands = [rgb[ya:yb, max(0, x0 - pad):max(0, x0)], rgb[ya:yb, min(rgb.shape[1], x1):min(rgb.shape[1], x1 + pad)]]
    px = np.concatenate([b.reshape(-1, 3) for b in bands if b.size])  if any(b.size for b in bands) else None
    return 0.0 if px is None else white_fraction(px.reshape(-1, 1, 3))


def decide(rgb: np.ndarray, person_boxes: list[Box], th: PersonThresholds = PersonThresholds()) -> PeopleDecision:
    height = rgb.shape[0]
    best = PeopleDecision(False)
    best_rank: tuple[bool, float] = (False, -1.0)
    for box in person_boxes:
        if (box[3] - box[1]) < th.min_height_ratio * height:
            continue
        # el blanco del fondo (cielo nublado, pared) no cuenta como gorra ni como uniforme
        head = max(0.0, white_fraction(_zone(rgb, box, 0.25, 0.0, 0.75, 0.12)) - 0.8 * _background_white(rgb, box, 0.0, 0.12))
        torso = max(0.0, white_fraction(_zone(rgb, box, 0.1, 0.2, 0.9, 0.55)) - 0.8 * _background_white(rgb, box, 0.2, 0.55))
        is_white = head >= th.head and torso >= th.torso
        rank = (is_white, min(head / th.head, torso / th.torso))
        if rank > best_rank:
            best, best_rank = PeopleDecision(is_white, head, torso), rank
    return best


class PersonFilter:
    def __init__(self, weights: str = "yolo11s.pt", device: str | None = None,
                 thresholds: PersonThresholds = PersonThresholds()):
        import torch
        from ultralytics import YOLO

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(weights)
        self.thresholds = thresholds

    def boxes(self, rgb: np.ndarray) -> list[Box]:
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        res = self.model.predict(bgr, classes=[0], conf=0.35, imgsz=1280, device=self.device, verbose=False)[0]
        return [tuple(int(v) for v in b) for b in res.boxes.xyxy.cpu().numpy()]

    def check(self, rgb: np.ndarray) -> PeopleDecision:
        return decide(rgb, self.boxes(rgb), self.thresholds)
