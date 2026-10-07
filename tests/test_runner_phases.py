import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from acfc.core.manifest import read_manifest, write_row
from acfc.core.runner import RunConfig, Runner


def _foto_con_mapa(path, w=400, h=600):
    img = np.full((h, w, 3), (90, 110, 70), dtype=np.uint8)
    img[h - 120:h, 0:120] = (197, 232, 197)
    cx, tip = 60, h - 60
    cv2.circle(img, (cx, tip - 15), 7, (234, 67, 53), -1)
    cv2.fillPoly(img, [np.array([[cx - 6, tip - 12], [cx + 6, tip - 12], [cx, tip]], np.int32)], (234, 67, 53))
    Image.fromarray(img).save(path)


def _run(cfg):
    return list(Runner(cfg).events())


def test_rutas_de_entrada_y_salida_por_fase(tmp_path):
    base = tmp_path
    for fase, (i, o) in {0: (base, base / "outputs"), 1: (base, base / "fase1_mapa"),
                          2: (base / "fase1_mapa", base / "fase2_personas"),
                          3: (base / "fase2_personas", base / "outputs")}.items():
        cfg = RunConfig(base, phase=fase)
        assert (cfg.in_dir, cfg.out_dir) == (i, o)


def test_fase1_escribe_manifiesto_y_logs(tmp_path, make_image):
    _foto_con_mapa(tmp_path / "m.jpg")
    make_image(tmp_path / "n.jpg", size=(400, 600))
    (tmp_path / "rota.jpg").write_bytes(b"basura")
    events = _run(RunConfig(tmp_path, phase=1, workers=2, backend="fake"))
    assert events[0]["type"] == "start" and events[0]["fase"] == 1 and events[0]["total"] == 3
    assert events[-1]["counts"] == {"cropped": 1, "copied": 1, "error": 1}
    out = tmp_path / "fase1_mapa"
    assert read_manifest(out) == {"m.jpg": ("abajo", pytest.approx(473, abs=8)), "n.jpg": ("ninguno", 0)}
    err = list(csv.DictReader((out / "log_fase1_errores.csv").open(encoding="utf-8")))
    assert [r["archivo"] for r in err] == ["rota.jpg"]


def test_fase2_pasa_el_manifiesto_solo_de_las_que_sobreviven(tmp_path, make_image):
    f1 = tmp_path / "fase1_mapa"
    f1.mkdir()
    make_image(f1 / "ok.jpg"); make_image(f1 / "blanca.jpg", color=(255, 0, 0))
    write_row(f1, "ok.jpg", "abajo", 480); write_row(f1, "blanca.jpg", "ninguno", 0)
    events = _run(RunConfig(tmp_path, phase=2, workers=1, backend="fake"))
    assert events[-1]["counts"] == {"copied": 1, "discarded": 1}
    out = tmp_path / "fase2_personas"
    assert read_manifest(out) == {"ok.jpg": ("abajo", 480)}
    desc = list(csv.DictReader((out / "log_fase2_descartadas.csv").open(encoding="utf-8")))
    assert [r["archivo"] for r in desc] == ["blanca.jpg"]
    assert not (out / "blanca.jpg").exists()


def test_fase3_usa_el_lado_del_manifiesto(tmp_path, make_image):
    f2 = tmp_path / "fase2_personas"
    f2.mkdir()
    make_image(f2 / "a.jpg", size=(100, 100)); make_image(f2 / "b.jpg", size=(100, 100))
    write_row(f2, "a.jpg", "abajo", 80)        # b.jpg no esta en el manifiesto: dos franjas
    events = _run(RunConfig(tmp_path, phase=3, workers=1, backend="fake"))
    assert events[-1]["counts"] == {"saved": 2}
    assert Image.open(tmp_path / "outputs" / "a.jpg").size == (100, 90)      # solo la caja de abajo
    assert Image.open(tmp_path / "outputs" / "b.jpg").size != (100, 90)      # ambas cajas: otro recorte


def test_carpeta_de_entrada_inexistente_da_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="falta la carpeta de la fase anterior"):
        _run(RunConfig(tmp_path, phase=2, backend="fake"))


def test_carpeta_de_entrada_vacia_termina_limpio(tmp_path):
    (tmp_path / "fase1_mapa").mkdir()
    events = _run(RunConfig(tmp_path, phase=2, workers=2, backend="fake"))
    assert [e["type"] for e in events] == ["start", "finished"] and events[-1]["counts"] == {}


def test_force_en_fase_reinicia_logs_y_manifiesto(tmp_path, make_image):
    f1 = tmp_path / "fase1_mapa"
    f1.mkdir()
    make_image(f1 / "blanca.jpg", color=(255, 0, 0))
    write_row(f1, "blanca.jpg", "ninguno", 0)
    _run(RunConfig(tmp_path, phase=2, workers=1, backend="fake"))
    _run(RunConfig(tmp_path, phase=2, workers=1, backend="fake", force=True))
    out = tmp_path / "fase2_personas"
    desc = list(csv.DictReader((out / "log_fase2_descartadas.csv").open(encoding="utf-8")))
    assert len(desc) == 1                                     # sin fila duplicada de la corrida anterior


def test_free_memory_no_falla_sin_torch_ni_gpu():
    from acfc.core.runner import free_memory
    free_memory()
