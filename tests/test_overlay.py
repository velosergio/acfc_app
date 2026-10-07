import cv2
import numpy as np

from acfc.core.overlay import OverlayParams, find_compass, find_minimap, text_boxes_from_ocr

P = OverlayParams()


def _pts(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# ---------- texto ----------

def test_texto_cadena_anclada_al_borde_superior_y_extendida():
    # imagen 1000x1000; bloque arriba a la derecha; la primera línea (y 0-30) no la vio el OCR
    results = [
        (_pts(700, 40, 990, 70), "Boyaca 2026", 0.8),
        (_pts(760, 80, 995, 110), "Samaca", 0.7),
        (_pts(100, 20, 300, 60), "hojas", 0.9),          # falso positivo a la izquierda
    ]
    boxes = text_boxes_from_ocr(results, 1000, 1000, 0, 300, True, P)
    (x0, y0, x1, y1), = boxes
    assert y0 == 0 and y1 == 110 + 45 and x1 == 1000   # 45 = 1.5 alturas de línea (30 px)
    assert 660 <= x0 <= 700


def test_texto_abajo_se_extiende_hasta_el_borde_inferior():
    results = [
        (_pts(700, 100, 990, 130), "Altitud 3198", 0.8),   # coordenadas locales a la franja (empieza en y=700)
        (_pts(720, 140, 995, 170), "Velocidad 0.0", 0.8),
    ]
    (x0, y0, x1, y1), = text_boxes_from_ocr(results, 1000, 1000, 700, 300, False, P)
    assert y1 == 1000 and y0 == 800 - 45


def test_texto_linea_aislada_o_confianza_baja_se_ignora():
    una = [(_pts(800, 10, 990, 40), "Altitud", 0.9)]
    assert text_boxes_from_ocr(una, 1000, 1000, 0, 300, True, P) == []
    baja = [(_pts(800, 10, 990, 40), "ab", 0.9), (_pts(800, 50, 990, 80), "cd", 0.1)]
    assert text_boxes_from_ocr(baja, 1000, 1000, 0, 300, True, P) == []
    assert text_boxes_from_ocr([], 1000, 1000, 0, 300, True, P) == []


def test_linea_debil_se_une_a_la_cadena_y_ensancha_el_bloque():
    # la línea de coordenadas (larga, números estilizados) sale con confianza muy baja
    results = [
        (_pts(300, 20, 990, 50), "6.0506N 74.4910W", 0.05),
        (_pts(900, 60, 995, 90), "Calle 11", 0.9),
        (_pts(800, 100, 995, 130), "Boyaca", 0.9),
    ]
    (x0, y0, x1, y1), = text_boxes_from_ocr(results, 1000, 1000, 0, 400, True, P)
    assert x0 <= 300 and y0 == 0


def test_cadena_sin_digitos_no_es_el_bloque_gps():
    # follaje que el OCR lee como palabras: el bloque real siempre trae fecha/coordenadas
    results = [
        (_pts(800, 10, 990, 40), "hoja verde", 0.8),
        (_pts(800, 60, 990, 90), "rama seca", 0.8),
    ]
    assert text_boxes_from_ocr(results, 1000, 1000, 0, 400, True, P) == []


def test_cadena_lejos_del_borde_no_es_el_bloque():
    # dos líneas fuertes pero a 35 % de la altura: no está pegado al borde, es ruido
    results = [
        (_pts(800, 350, 990, 380), "Boyaca 2026", 0.9),
        (_pts(800, 390, 990, 420), "Samaca 2026", 0.9),
    ]
    assert text_boxes_from_ocr(results, 1000, 1000, 0, 450, True, P) == []


def test_texto_lejos_del_borde_no_encadena():
    results = [
        (_pts(800, 10, 990, 40), "Boyaca 2026", 0.8),
        (_pts(800, 60, 990, 90), "Samaca 2026", 0.8),
        (_pts(800, 280, 990, 310), "arbol", 0.8),        # demasiado lejos de la cadena
    ]
    (x0, y0, x1, y1), = text_boxes_from_ocr(results, 1000, 1000, 0, 300, True, P)
    assert y1 == 90 + 45


# ---------- minimapa (pin rojo en el centro) ----------

def _fondo(w=400, h=600):
    return np.full((h, w, 3), (90, 110, 70), dtype=np.uint8)


def _tile(img, x0, y0, x1, y1):
    img[y0:y1, x0:x1] = (197, 232, 197)
    cx, tip = (x0 + x1) // 2, (y0 + y1) // 2
    # pin: cabeza redonda + punta, 14 px de ancho x 22 de alto, punta en el centro del mosaico
    cv2.circle(img, (cx, tip - 15), 7, (234, 67, 53), -1)
    pts = np.array([[cx - 6, tip - 12], [cx + 6, tip - 12], [cx, tip]], np.int32)
    cv2.fillPoly(img, [pts], (234, 67, 53))


def test_minimapa_arriba_izquierda():
    img = _fondo()
    _tile(img, 0, 0, 120, 120)
    (x0, y0, x1, y1), = find_minimap(img, P)
    assert (x0, y0) == (0, 0)
    assert abs(x1 - 120) <= 6 and abs(y1 - 120) <= 6


def test_minimapa_abajo_derecha():
    img = _fondo()
    _tile(img, 280, 480, 400, 600)
    (x0, y0, x1, y1), = find_minimap(img, P)
    assert (x1, y1) == (400, 600)
    assert abs(x0 - 280) <= 6 and abs(y0 - 480) <= 6


def test_sin_pin_no_hay_minimapa():
    img = _fondo()
    img[0:120, 0:120] = (197, 232, 197)
    assert find_minimap(img, P) == []
    assert find_minimap(np.full((600, 400, 3), 225, dtype=np.uint8), P) == []


def test_objeto_rojo_sin_mapa_no_es_minimapa():
    img = _fondo()
    cv2.circle(img, (60, 45), 7, (234, 67, 53), -1)  # pin sobre fondo que no es mapa
    pts = np.array([[54, 48], [66, 48], [60, 60]], np.int32)
    cv2.fillPoly(img, [pts], (234, 67, 53))
    assert find_minimap(img, P) == []


# ---------- brújula (aguja cian) ----------

def _brujula(img, x0, y0, size):
    cv2.circle(img, (x0 + size // 2, y0 + size // 2), size // 2 - 2, (70, 70, 70), 6)
    pts = np.array([[x0 + size // 2, y0 + 15], [x0 + size // 2 - 8, y0 + size // 2], [x0 + size // 2 + 8, y0 + size // 2]], np.int32)
    cv2.fillPoly(img, [pts], (0, 190, 220))


def test_brujula_arriba_izquierda():
    img = _fondo()
    _brujula(img, 0, 0, 84)
    (x0, y0, x1, y1), = find_compass(img, P)
    assert (x0, y0) == (0, 0) and abs(x1 - 84) <= 6 and abs(y1 - 84) <= 6


def test_brujula_abajo_izquierda():
    img = _fondo()
    _brujula(img, 0, 516, 84)
    (x0, y0, x1, y1), = find_compass(img, P)
    assert x0 == 0 and y1 == 600 and abs(y0 - 516) <= 6


def test_sin_brujula():
    assert find_compass(_fondo(), P) == []
