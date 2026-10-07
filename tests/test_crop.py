from acfc.core.crop import largest_free_rect, pad_box


def test_sin_cajas_devuelve_imagen_completa():
    assert largest_free_rect(100, 100, []) == (0, 0, 100, 100)


def test_una_caja_en_esquina():
    # caja arriba-izquierda: la mayor franja libre es la derecha (80x100)
    assert largest_free_rect(100, 100, [(0, 0, 20, 50)]) == (20, 0, 100, 100)


def test_mapa_abajo_izquierda_y_texto_arriba_derecha():
    boxes = [(0, 60, 30, 100), (60, 0, 100, 40)]
    assert largest_free_rect(100, 100, boxes) == (30, 40, 100, 100)


def test_cajas_cubren_todo_devuelve_none():
    assert largest_free_rect(100, 100, [(0, 0, 100, 100)]) is None


def test_pad_box_se_recorta_a_los_limites():
    assert pad_box((5, 5, 95, 95), 100, 100, 0.1) == (0, 0, 100, 100)
    assert pad_box((40, 40, 50, 50), 100, 100, 0.05) == (35, 35, 55, 55)
