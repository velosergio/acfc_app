from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def make_image():
    def _make(path, size=(100, 100), color=(10, 20, 30), mode="RGB"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, size, color).save(path)
        return path

    return _make
