from __future__ import annotations

from acfc.core.crop import Box
from acfc.core.people import PeopleDecision


class FakePeople:
    def check(self, rgb) -> PeopleDecision:
        r, g, b = (int(v) for v in rgb[0, 0])  # tolerante: JPEG altera el rojo puro
        return PeopleDecision(discard=r > 230 and g < 40 and b < 40)


class FakeOverlay:
    last_sides: tuple = ()

    def detect(self, rgb) -> list[Box]:
        h, w = rgb.shape[:2]
        return [(0, 0, w // 5, h // 10)]

    def text_boxes(self, rgb, sides) -> list[Box]:
        self.last_sides = tuple(sides)
        h, w = rgb.shape[:2]
        boxes = []
        if "arriba" in sides:
            boxes.append((0, 0, w // 5, h // 10))
        if "abajo" in sides:
            boxes.append((w - w // 5, h - h // 10, w, h))
        return boxes


def build_backend(name: str):
    if name == "fake":
        return FakePeople(), FakeOverlay()
    from acfc.core.overlay import OverlayDetector
    from acfc.core.people import PersonFilter

    return PersonFilter(), OverlayDetector()
