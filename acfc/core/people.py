from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from acfc.core.crop import Box


@dataclass(frozen=True)
class PersonThresholds:
    head: float = 0.35
    torso: float = 0.40
    min_height_ratio: float = 0.12


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


def decide(rgb: np.ndarray, person_boxes: list[Box], th: PersonThresholds = PersonThresholds()) -> PeopleDecision:
    height = rgb.shape[0]
    best = PeopleDecision(False)
    best_rank: tuple[bool, float] = (False, -1.0)
    for box in person_boxes:
        if (box[3] - box[1]) < th.min_height_ratio * height:
            continue
        head = white_fraction(_zone(rgb, box, 0.2, 0.0, 0.8, 0.15))
        torso = white_fraction(_zone(rgb, box, 0.1, 0.2, 0.9, 0.55))
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

    def check(self, rgb: np.ndarray) -> PeopleDecision:
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        res = self.model.predict(bgr, classes=[0], conf=0.35, device=self.device,
                                 half=self.device == "cuda", verbose=False)[0]
        boxes = [tuple(int(v) for v in b) for b in res.boxes.xyxy.cpu().numpy()]
        return decide(rgb, boxes, self.thresholds)
