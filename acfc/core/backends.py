from __future__ import annotations

from acfc.core.crop import Box
from acfc.core.people import PeopleDecision


class FakePeople:
    def check(self, rgb) -> PeopleDecision:
        r, g, b = (int(v) for v in rgb[0, 0])  # tolerante: JPEG altera el rojo puro
        return PeopleDecision(discard=r > 230 and g < 40 and b < 40)


class FakeOverlay:
    def detect(self, rgb) -> list[Box]:
        h, w = rgb.shape[:2]
        return [(0, 0, w // 5, h // 10)]


def build_backend(name: str):
    if name == "fake":
        return FakePeople(), FakeOverlay()
    from acfc.core.overlay import OverlayDetector
    from acfc.core.people import PersonFilter

    return PersonFilter(), OverlayDetector()
