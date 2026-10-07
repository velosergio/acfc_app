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


from acfc.core.backends import build_backend
from acfc.core.phases import phase2_people, phase3_text


def test_fase2_descarta_y_no_copia(tmp_path, make_image):
    src = make_image(tmp_path / "w.jpg", color=(255, 0, 0))
    out = tmp_path / "f2"; out.mkdir()
    people, _ = build_backend("fake")
    r = phase2_people(src, out, people)
    assert r.status == "discarded" and not (out / "w.jpg").exists()


def test_fase2_copia_sin_recomprimir_a_las_que_pasan(tmp_path, make_image):
    src = make_image(tmp_path / "ok.jpg")
    out = tmp_path / "f2"; out.mkdir()
    people, _ = build_backend("fake")
    r = phase2_people(src, out, people)
    assert r.status == "copied"
    assert (out / "ok.jpg").read_bytes() == src.read_bytes()


def test_fase2_omite_force_y_error(tmp_path, make_image):
    src = make_image(tmp_path / "ok.jpg")
    out = tmp_path / "f2"; out.mkdir()
    people, _ = build_backend("fake")
    assert phase2_people(src, out, people).status == "copied"
    assert phase2_people(src, out, people).status == "skipped"
    assert phase2_people(src, out, people, force=True).status == "copied"
    bad = tmp_path / "bad.jpg"; bad.write_bytes(b"basura")
    assert phase2_people(bad, out, people).status == "error"


def test_fase3_solo_mira_el_lado_del_manifiesto(tmp_path, make_image):
    src = make_image(tmp_path / "a.jpg", size=(100, 100))
    out = tmp_path / "out"; out.mkdir()
    _, overlay = build_backend("fake")
    r = phase3_text(src, out, overlay, side="abajo")
    assert overlay.last_sides == ("abajo",)
    assert r.status == "saved" and Image.open(out / "a.jpg").size == (100, 90)   # recorta la caja de abajo


def test_fase3_sin_lado_o_ninguno_mira_las_dos_franjas(tmp_path, make_image):
    src = make_image(tmp_path / "a.jpg", size=(100, 100))
    out = tmp_path / "out"; out.mkdir()
    _, overlay = build_backend("fake")
    phase3_text(src, out, overlay)                      # sin manifiesto: lado por defecto
    assert overlay.last_sides == ("arriba", "abajo")
    phase3_text(src, out, overlay, side="ninguno", force=True)
    assert overlay.last_sides == ("arriba", "abajo")


def test_fase3_sin_texto_copia_los_bytes(tmp_path, make_image):
    src = make_image(tmp_path / "a.jpg", size=(100, 100))
    out = tmp_path / "out"; out.mkdir()

    class SinTexto:
        def text_boxes(self, rgb, sides):
            return []

    r = phase3_text(src, out, SinTexto())
    assert r.status == "saved" and (out / "a.jpg").read_bytes() == src.read_bytes()


def test_fase3_cajas_cubren_todo_guarda_original_y_revisa(tmp_path, make_image):
    src = make_image(tmp_path / "a.jpg", size=(100, 100))
    out = tmp_path / "out"; out.mkdir()

    class Todo:
        def text_boxes(self, rgb, sides):
            return [(0, 0, 100, 100)]

    r = phase3_text(src, out, Todo())
    assert r.status == "review" and Image.open(out / "a.jpg").size == (100, 100)


def test_fase3_recorte_pequeno_marca_revisar(tmp_path, make_image):
    src = make_image(tmp_path / "a.jpg", size=(100, 100))
    out = tmp_path / "out"; out.mkdir()

    class Grande:
        def text_boxes(self, rgb, sides):
            return [(0, 0, 100, 70)]

    r = phase3_text(src, out, Grande(), min_area_ratio=0.4)
    assert r.status == "review" and r.area_ratio < 0.4 and (out / "a.jpg").exists()


def test_fase3_respeta_la_orientacion_exif(tmp_path):
    # 100x60 con orientacion 6 (girar 90): la imagen real es 60x100 y el recorte debe hacerse sobre eso
    p = tmp_path / "r.jpg"
    exif = Image.Exif(); exif[0x0112] = 6
    Image.new("RGB", (100, 60), (10, 20, 30)).save(p, exif=exif)
    out = tmp_path / "out"; out.mkdir()
    _, overlay = build_backend("fake")
    phase3_text(p, out, overlay, side="arriba")         # caja (0,0,12,10) sobre 60x100
    assert Image.open(out / "r.jpg").size == (60, 90)


def test_fase3_omite_force_y_error(tmp_path, make_image):
    src = make_image(tmp_path / "a.jpg", size=(100, 100))
    out = tmp_path / "out"; out.mkdir()
    _, overlay = build_backend("fake")
    assert phase3_text(src, out, overlay).status == "saved"
    assert phase3_text(src, out, overlay).status == "skipped"
    assert phase3_text(src, out, overlay, force=True).status == "saved"
    bad = tmp_path / "bad.jpg"; bad.write_bytes(b"basura")
    assert phase3_text(bad, out, overlay).status == "error"
