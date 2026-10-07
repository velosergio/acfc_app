import numpy as np

from acfc.core.people import PersonThresholds, decide, white_fraction


def _img():
    return np.full((400, 200, 3), 60, dtype=np.uint8)


def _paint(img, box, fx0, fy0, fx1, fy1, value=255):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    img[int(y0 + fy0 * h):int(y0 + fy1 * h), int(x0 + fx0 * w):int(x0 + fx1 * w)] = value


def test_white_fraction():
    white = np.full((10, 10, 3), 255, dtype=np.uint8)
    dark = np.full((10, 10, 3), 40, dtype=np.uint8)
    assert white_fraction(white) == 1.0
    assert white_fraction(dark) == 0.0
    assert white_fraction(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


def test_gorra_y_torso_blancos_descarta():
    img, box = _img(), (50, 100, 150, 350)
    _paint(img, box, 0.2, 0.0, 0.8, 0.15)   # gorra
    _paint(img, box, 0.1, 0.2, 0.9, 0.55)   # torso
    assert decide(img, [box]).discard is True


def test_solo_torso_blanco_no_descarta():
    img, box = _img(), (50, 100, 150, 350)
    _paint(img, box, 0.1, 0.2, 0.9, 0.55)
    assert decide(img, [box]).discard is False


def test_solo_gorra_blanca_no_descarta():
    img, box = _img(), (50, 100, 150, 350)
    _paint(img, box, 0.2, 0.0, 0.8, 0.15)
    assert decide(img, [box]).discard is False


def test_persona_muy_pequena_se_ignora():
    img, box = _img(), (50, 100, 70, 130)  # alto 30 < 12 % de 400
    _paint(img, box, 0.0, 0.0, 1.0, 1.0)
    assert decide(img, [box], PersonThresholds()).discard is False


def test_cielo_blanco_detras_no_cuenta_como_gorra():
    img = np.full((400, 300, 3), 255, dtype=np.uint8)      # fondo blanco (cielo nublado)
    box = (100, 100, 200, 350)
    _paint(img, box, 0.1, 0.2, 0.9, 0.55, value=250)       # torso "blanco" tambien
    assert decide(img, [box]).discard is False


def test_gorra_blanca_con_fondo_oscuro_descarta_aunque_ocupe_poco():
    img, box = _img(), (50, 100, 150, 350)
    _paint(img, box, 0.25, 0.0, 0.55, 0.12)   # gorra: solo ~40 % de la zona de cabeza
    _paint(img, box, 0.1, 0.2, 0.9, 0.55)
    assert decide(img, [box]).discard is True


def test_una_persona_blanca_entre_varias_descarta():
    img = _img()
    a, b = (10, 100, 80, 350), (110, 100, 190, 350)
    _paint(img, b, 0.2, 0.0, 0.8, 0.15)
    _paint(img, b, 0.1, 0.2, 0.9, 0.55)
    assert decide(img, [a, b]).discard is True
