import numpy as np

from acfc.core.overlay import OverlayParams, find_minimap, text_boxes_from_ocr

P = OverlayParams()


def _pts(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_texto_une_solo_lineas_alineadas_a_la_derecha():
    results = [
        (_pts(700, 10, 990, 40), "Local: 29/05/2026", 0.8),
        (_pts(800, 50, 995, 80), "Boyaca", 0.7),
        (_pts(100, 20, 300, 60), "hojas", 0.9),          # falso positivo a la izquierda
        (_pts(700, 90, 990, 120), "ruido", 0.05),        # confianza muy baja
    ]
    assert text_boxes_from_ocr(results, 1000, 0, P) == [(700, 10, 995, 80)]


def test_texto_aplica_offset_vertical_y_sin_resultados():
    results = [(_pts(800, 5, 990, 30), "Altitud", 0.9)]
    assert text_boxes_from_ocr(results, 1000, 500, P) == [(800, 505, 990, 530)]
    assert text_boxes_from_ocr([], 1000, 0, P) == []


def _fondo():
    return np.full((600, 400, 3), (90, 110, 70), dtype=np.uint8)


def test_minimapa_arriba_izquierda():
    img = _fondo()
    img[0:90, 0:120] = (197, 232, 197)
    (x0, y0, x1, y1), = find_minimap(img, P)
    assert (x0, y0) == (0, 0)
    assert abs(x1 - 120) <= 12 and abs(y1 - 90) <= 12


def test_minimapa_abajo_derecha():
    img = _fondo()
    img[510:600, 280:400] = (197, 232, 197)
    (x0, y0, x1, y1), = find_minimap(img, P)
    assert (x1, y1) == (400, 600)
    assert abs(x0 - 280) <= 12 and abs(y0 - 510) <= 12


def test_sin_minimapa_en_fondo_uniforme():
    assert find_minimap(_fondo(), P) == []


def test_cielo_nublado_no_es_minimapa():
    img = np.full((600, 400, 3), 225, dtype=np.uint8)
    assert find_minimap(img, P) == []
