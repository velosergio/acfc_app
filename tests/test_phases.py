from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from acfc.core.phases import phase1_map


def _foto_con_mapa(path, w=400, h=600, arriba=False):
    img = np.full((h, w, 3), (90, 110, 70), dtype=np.uint8)
    y0, y1 = (0, 120) if arriba else (h - 120, h)
    img[y0:y1, 0:120] = (197, 232, 197)
    cx, tip = 60, (y0 + y1) // 2
    cv2.circle(img, (cx, tip - 15), 7, (234, 67, 53), -1)
    cv2.fillPoly(img, [np.array([[cx - 6, tip - 12], [cx + 6, tip - 12], [cx, tip]], np.int32)], (234, 67, 53))
    Image.fromarray(img).save(path)
    return Path(path)


def test_fase1_mapa_abajo_corta_la_franja_de_abajo(tmp_path):
    src = _foto_con_mapa(tmp_path / "a.jpg")
    out = tmp_path / "f1"; out.mkdir()
    r = phase1_map(src, out)
    assert r.status == "cropped" and r.side == "abajo"
    w, h = Image.open(out / "a.jpg").size
    assert w == 400 and abs(h - (480 - 7)) <= 8        # alto 600 - mapa 120 - margen int(0.012*600)=7
    assert abs(r.cut - h) <= 1


def test_fase1_mapa_arriba_conserva_lo_de_abajo(tmp_path):
    src = _foto_con_mapa(tmp_path / "t.jpg", arriba=True)
    out = tmp_path / "f1"; out.mkdir()
    r = phase1_map(src, out)
    assert r.status == "cropped" and r.side == "arriba"
    w, h = Image.open(out / "t.jpg").size
    assert w == 400 and abs(h - (480 - 7)) <= 8
    assert abs(r.cut - (120 + 7)) <= 8                  # filas retiradas por arriba


def test_fase1_sin_mapa_copia_los_bytes_sin_recomprimir(tmp_path, make_image):
    src = make_image(tmp_path / "n.jpg", size=(400, 600))
    out = tmp_path / "f1"; out.mkdir()
    r = phase1_map(src, out)
    assert r.status == "copied" and r.side == "ninguno" and r.cut == 0
    assert (out / "n.jpg").read_bytes() == src.read_bytes()


def test_fase1_corte_grande_marca_revisar_pero_guarda(tmp_path):
    src = _foto_con_mapa(tmp_path / "a.jpg")
    out = tmp_path / "f1"; out.mkdir()
    r = phase1_map(src, out, min_keep=0.9)              # queda ~79 %: por debajo del 90 %
    assert r.status == "review" and (out / "a.jpg").exists()


def test_fase1_omite_si_existe_y_force_rehace(tmp_path):
    src = _foto_con_mapa(tmp_path / "a.jpg")
    out = tmp_path / "f1"; out.mkdir()
    assert phase1_map(src, out).status == "cropped"
    assert phase1_map(src, out).status == "skipped"
    assert phase1_map(src, out, force=True).status == "cropped"


def test_fase1_archivo_corrupto_es_error_y_no_deja_tmp(tmp_path):
    bad = tmp_path / "bad.jpg"; bad.write_bytes(b"basura")
    out = tmp_path / "f1"; out.mkdir()
    r = phase1_map(bad, out)
    assert r.status == "error" and r.reason
    assert not list(out.glob(".*.tmp"))
