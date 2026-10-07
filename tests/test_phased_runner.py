from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from acfc.core.runner import PhasedRunner, RunConfig, Runner, make_runner


def _foto_con_mapa(path, w=400, h=600):
    img = np.full((h, w, 3), (90, 110, 70), dtype=np.uint8)
    img[h - 120:h, 0:120] = (197, 232, 197)
    cx, tip = 60, h - 60
    cv2.circle(img, (cx, tip - 15), 7, (234, 67, 53), -1)
    cv2.fillPoly(img, [np.array([[cx - 6, tip - 12], [cx + 6, tip - 12], [cx, tip]], np.int32)], (234, 67, 53))
    Image.fromarray(img).save(path)


def test_make_runner_segun_el_modo(tmp_path):
    assert isinstance(make_runner(RunConfig(tmp_path, mode="fases")), PhasedRunner)
    assert type(make_runner(RunConfig(tmp_path, mode="unico"))) is Runner


def test_tres_fases_encadenadas_con_backend_fake(tmp_path, make_image):
    _foto_con_mapa(tmp_path / "mapa.jpg")                    # fase 1 corta abajo; fase 3 mira solo abajo
    make_image(tmp_path / "sin.jpg", size=(100, 100))
    make_image(tmp_path / "blanca.jpg", size=(100, 100), color=(255, 0, 0))
    events = list(PhasedRunner(RunConfig(tmp_path, mode="fases", workers=2, backend="fake")).events())

    tipos = [(e["type"], e.get("fase")) for e in events if e["type"] in ("start", "phase_done")]
    assert tipos == [("start", 1), ("phase_done", 1), ("start", 2), ("phase_done", 2), ("start", 3), ("phase_done", 3)]
    assert events[-1]["type"] == "finished" and events[-1]["cancelled"] is False
    assert events[-1]["phases"][2] == {"copied": 2, "discarded": 1}
    assert events[-1]["counts"] == events[-1]["phases"][3] == {"saved": 2}

    assert not (tmp_path / "fase2_personas" / "blanca.jpg").exists()
    assert not (tmp_path / "outputs" / "blanca.jpg").exists()
    w, h = Image.open(tmp_path / "outputs" / "mapa.jpg").size
    f1_h = Image.open(tmp_path / "fase1_mapa" / "mapa.jpg").size[1]
    assert w == 400 and h == f1_h - f1_h // 10               # fase 3: solo la caja de abajo del OCR falso


def test_se_puede_lanzar_una_sola_fase(tmp_path, make_image):
    make_image(tmp_path / "a.jpg", size=(100, 100))
    events = list(PhasedRunner(RunConfig(tmp_path, mode="fases", phases=(1,), backend="fake")).events())
    assert [e["fase"] for e in events if e["type"] == "start"] == [1]
    assert (tmp_path / "fase1_mapa" / "a.jpg").exists() and not (tmp_path / "outputs").exists()


def test_cancelar_antes_de_empezar_no_ejecuta_fases_ni_deja_tmp(tmp_path, make_image):
    for i in range(5):
        make_image(tmp_path / f"i{i}.jpg", size=(100, 100))
    runner = PhasedRunner(RunConfig(tmp_path, mode="fases", workers=2, backend="fake"))
    runner.cancel()
    events = list(runner.events())
    assert events[-1]["type"] == "finished" and events[-1]["cancelled"] is True
    assert not [e for e in events if e["type"] == "start"]
    assert not list(tmp_path.rglob(".*.tmp"))
