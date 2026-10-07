import csv
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image

from acfc.cli import main
from acfc.core.phases import phase4_size
from acfc.core.runner import PhasedRunner, RunConfig, Runner
from acfc.core.timing import eta_seconds, fmt_hms


# ---------- tiempo y ETA ----------

def test_formato_hms():
    assert fmt_hms(0) == "00:00:00"
    assert fmt_hms(3725.4) == "01:02:05"
    assert fmt_hms(None) == "--:--:--"


def test_eta_se_calcula_con_el_ritmo_observado():
    assert eta_seconds(elapsed=10, done=5, total=25) == pytest.approx(40)
    assert eta_seconds(elapsed=10, done=25, total=25) == 0
    assert eta_seconds(elapsed=10, done=0, total=25) is None          # aun sin ritmo medible


# ---------- fase 4: tamano minimo ----------

def test_fase4_elimina_si_ancho_o_alto_menor_al_minimo(tmp_path, make_image):
    chica_ancho = make_image(tmp_path / "a.jpg", size=(499, 800))
    chica_alto = make_image(tmp_path / "b.jpg", size=(800, 499))
    justa = make_image(tmp_path / "c.jpg", size=(500, 500))
    grande = make_image(tmp_path / "d.jpg", size=(900, 700))
    assert phase4_size(chica_ancho, 500).status == "discarded" and not chica_ancho.exists()
    r = phase4_size(chica_alto, 500)
    assert r.status == "discarded" and "499" in r.reason and not chica_alto.exists()
    assert phase4_size(justa, 500).status == "saved" and justa.exists()
    assert phase4_size(grande, 500).status == "saved" and grande.exists()


def test_fase4_archivo_corrupto_es_error_y_no_se_borra(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"basura")
    assert phase4_size(bad, 500).status == "error" and bad.exists()


def test_fase4_en_el_runner_borra_de_outputs_y_deja_log(tmp_path, make_image):
    out = tmp_path / "outputs"
    make_image(out / "chica.jpg", size=(300, 900))
    make_image(out / "ok.jpg", size=(800, 900))
    events = list(Runner(RunConfig(tmp_path, phase=4, workers=1, backend="fake", min_side=500)).events())
    assert events[-1]["counts"] == {"discarded": 1, "saved": 1}
    assert not (out / "chica.jpg").exists() and (out / "ok.jpg").exists()
    rows = list(csv.DictReader((out / "log_fase4_descartadas.csv").open(encoding="utf-8")))
    assert [r["archivo"] for r in rows] == ["chica.jpg"]


def test_fase4_sin_carpeta_outputs_da_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="falta la carpeta de la fase anterior"):
        list(Runner(RunConfig(tmp_path, phase=4, backend="fake")).events())


def test_fases_1_a_4_encadenadas_descartan_las_pequenas(tmp_path, make_image):
    make_image(tmp_path / "chica.jpg", size=(300, 300))
    make_image(tmp_path / "grande.jpg", size=(900, 900))
    cfg = RunConfig(tmp_path, mode="fases", phases=(1, 2, 3, 4), workers=1, backend="fake", min_side=500)
    events = list(PhasedRunner(cfg).events())
    assert [e["fase"] for e in events if e["type"] == "start"] == [1, 2, 3, 4]
    # el OCR falso recorta una caja arriba (fase 3): la grande queda en 900 x 810, mayor que 500
    assert (tmp_path / "outputs" / "grande.jpg").exists() and not (tmp_path / "outputs" / "chica.jpg").exists()


def test_cli_todas_incluye_fase_4_y_acepta_min_lado(tmp_path, make_image):
    make_image(tmp_path / "chica.jpg", size=(300, 300))
    make_image(tmp_path / "grande.jpg", size=(900, 900))
    assert main([str(tmp_path), "--backend", "fake", "--workers", "1", "--min-lado", "500"]) == 0
    assert [p.name for p in (tmp_path / "outputs").glob("*.jpg")] == ["grande.jpg"]


def test_cli_sin_personas_mantiene_la_fase_4(tmp_path, make_image):
    make_image(tmp_path / "chica.jpg", size=(300, 300))
    assert main([str(tmp_path), "--sin-personas", "--backend", "fake", "--workers", "1"]) == 0
    assert not (tmp_path / "fase2_personas").exists() and not list((tmp_path / "outputs").glob("*.jpg"))


# ---------- GUI: contador de tiempo y ETA ----------

def test_gui_muestra_tiempo_transcurrido_y_eta():
    from PySide6.QtWidgets import QApplication

    from acfc.gui import MainWindow
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    reloj = {"t": 100.0}
    win._now = lambda: reloj["t"]
    win.begin_timing()
    win.handle_event({"type": "start", "fase": 1, "total": 10, "workers": 1, "batches": 2})
    reloj["t"] = 110.0
    for i in range(5):
        win.handle_event({"type": "image_done", "worker": 0, "name": f"{i}.jpg", "status": "saved", "reason": "",
                          "head": 0.0, "torso": 0.0, "area_ratio": 1.0, "fase": 1})
    win.refresh_time()
    txt = win.time_label.text()
    assert "00:00:10" in txt and "ETA" in txt and "00:00:10" in txt.split("ETA")[1]    # 5 en 10 s: faltan 10 s


def test_gui_selector_incluye_fase_4_y_lado_minimo(tmp_path):
    from PySide6.QtWidgets import QApplication

    from acfc.gui import MainWindow
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.path_edit.setText(str(tmp_path))
    assert win.min_side_spin.value() == 500
    win.mode_combo.setCurrentIndex(0)
    assert win.build_config().phases == (1, 2, 3, 4)
    win.mode_combo.setCurrentIndex(1)
    cfg = win.build_config()
    assert (cfg.phases, cfg.skip_people) == ((1, 3, 4), True)
    win.mode_combo.setCurrentIndex(5)
    assert win.build_config().phases == (4,)
    win.min_side_spin.setValue(600)
    assert win.build_config().min_side == 600
