from acfc.core.manifest import read_manifest, write_row
from acfc.core.pipeline import ImageResult


def test_manifiesto_ida_y_vuelta(tmp_path):
    write_row(tmp_path, "a.jpg", "abajo", 480)
    write_row(tmp_path, "b con tilde ñ.jpg", "ninguno", 0)
    assert read_manifest(tmp_path) == {"a.jpg": ("abajo", 480), "b con tilde ñ.jpg": ("ninguno", 0)}
    assert (tmp_path / "manifest.csv").read_text(encoding="utf-8").splitlines()[0] == "archivo,lado,corte"


def test_manifiesto_inexistente_es_vacio(tmp_path):
    assert read_manifest(tmp_path / "nada") == {}


def test_image_result_tiene_lado_y_corte_por_defecto():
    r = ImageResult("a.jpg", "saved")
    assert (r.side, r.cut) == ("", 0)


def test_manifiesto_con_filas_danadas_las_ignora(tmp_path):
    write_row(tmp_path, "ok.jpg", "abajo", 10)
    with (tmp_path / "manifest.csv").open("a", encoding="utf-8") as fh:
        fh.write("trunca.jpg,abajo\nmala.jpg,abajo,xx\n")
    assert read_manifest(tmp_path) == {"ok.jpg": ("abajo", 10)}
