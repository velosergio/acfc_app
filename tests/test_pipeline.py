from pathlib import Path

from PIL import Image

from acfc.core.backends import build_backend
from acfc.core.pipeline import process_image


def _run(path, out, **kw):
    people, overlay = build_backend("fake")
    return process_image(Path(path), out, people, overlay, **kw)


def test_guarda_recorte_con_mismo_nombre(tmp_path, make_image):
    src = make_image(tmp_path / "a.jpg")
    out = tmp_path / "outputs"
    out.mkdir()
    res = _run(src, out)
    assert res.status == "saved"
    assert Image.open(out / "a.jpg").size == (100, 90)
    assert not list(out.glob(".*.tmp"))


def test_descarta_y_no_copia(tmp_path, make_image):
    src = make_image(tmp_path / "w.jpg", color=(255, 0, 0))
    out = tmp_path / "outputs"
    out.mkdir()
    res = _run(src, out)
    assert res.status == "discarded"
    assert not (out / "w.jpg").exists()


def test_omite_si_existe_y_force_rehace(tmp_path, make_image):
    src = make_image(tmp_path / "a.jpg")
    out = tmp_path / "outputs"
    out.mkdir()
    assert _run(src, out).status == "saved"
    assert _run(src, out).status == "skipped"
    assert _run(src, out, force=True).status == "saved"


def test_archivo_corrupto_es_error(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"no soy una imagen")
    out = tmp_path / "outputs"
    out.mkdir()
    res = _run(bad, out)
    assert res.status == "error" and res.reason


def test_modos_gris_alfa_y_cmyk(tmp_path, make_image):
    out = tmp_path / "outputs"
    out.mkdir()
    gris = make_image(tmp_path / "g.png", color=128, mode="L")
    alfa = make_image(tmp_path / "r.png", color=(1, 2, 3, 128), mode="RGBA")
    cmyk = make_image(tmp_path / "c.jpg", color=(10, 20, 30, 40), mode="CMYK")
    for p in (gris, alfa, cmyk):
        assert _run(p, out).status == "saved", p.name


def test_nombre_con_tildes_y_espacios(tmp_path, make_image):
    src = make_image(tmp_path / "Camión señal 1.jpeg")
    out = tmp_path / "outputs"
    out.mkdir()
    assert _run(src, out).status == "saved"
    assert (out / "Camión señal 1.jpeg").exists()


def test_cajas_cubren_todo_guarda_original_y_marca_revisar(tmp_path, make_image):
    src = make_image(tmp_path / "a.jpg")
    out = tmp_path / "outputs"
    out.mkdir()
    people, _ = build_backend("fake")

    class Todo:
        def detect(self, rgb):
            return [(0, 0, rgb.shape[1], rgb.shape[0])]

    res = process_image(src, out, people, Todo())
    assert res.status == "review"
    assert Image.open(out / "a.jpg").size == (100, 100)


def test_recorte_pequeno_marca_revisar(tmp_path, make_image):
    src = make_image(tmp_path / "a.jpg")
    out = tmp_path / "outputs"
    out.mkdir()
    people, _ = build_backend("fake")

    class Grande:
        def detect(self, rgb):
            return [(0, 0, 100, 70)]  # queda 30 % de área

    res = process_image(src, out, people, Grande(), min_area_ratio=0.4)
    assert res.status == "review" and res.area_ratio < 0.4
    assert (out / "a.jpg").exists()
