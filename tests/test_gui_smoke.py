import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from acfc.gui import MainWindow


def test_casilla_rehacer_existentes_llega_a_la_configuracion(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    assert win.force_check.isChecked() is False
    win.path_edit.setText(str(tmp_path))
    win.workers_spin.setValue(1)
    win.force_check.setChecked(True)
    cfg = win.build_config()
    assert cfg.force is True and cfg.workers == 1 and cfg.input_dir == tmp_path


def test_handle_event_actualiza_barras_y_tabla():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.handle_event({"type": "start", "total": 3, "workers": 2, "batches": 1})
    assert win.overall.maximum() == 3 and len(win.worker_bars) == 2
    win.handle_event({"type": "batch_start", "worker": 0, "size": 3, "batch": 1})
    assert win.worker_bars[0].maximum() == 3
    win.handle_event({"type": "image_done", "worker": 0, "name": "a.jpg", "status": "saved",
                      "reason": "", "head": 0.0, "torso": 0.0, "area_ratio": 0.8})
    win.handle_event({"type": "image_done", "worker": 0, "name": "b.jpg", "status": "discarded",
                      "reason": "uniforme y gorra blancos", "head": 0.5, "torso": 0.6, "area_ratio": 1.0})
    assert win.overall.value() == 2 and win.worker_bars[0].value() == 2
    assert win.table.rowCount() == 1 and win.table.item(0, 0).text() == "b.jpg"
    win.handle_event({"type": "finished", "counts": {"saved": 1, "discarded": 1}, "cancelled": False})
    assert win.start_btn.isEnabled() and not win.cancel_btn.isEnabled()


def test_selector_de_modo_llega_a_la_configuracion(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.path_edit.setText(str(tmp_path))
    assert win.mode_combo.count() == 7
    win.mode_combo.setCurrentIndex(0)
    cfg = win.build_config()
    assert (cfg.mode, cfg.phases, cfg.skip_people) == ("fases", (1, 2, 3, 4), False)
    win.mode_combo.setCurrentIndex(1)                              # Fases 1, 3 y 4 (sin personas)
    cfg = win.build_config()
    assert (cfg.mode, cfg.phases, cfg.skip_people) == ("fases", (1, 3, 4), True)
    win.mode_combo.setCurrentIndex(3)                              # Solo fase 2
    cfg = win.build_config()
    assert (cfg.mode, cfg.phases) == ("fases", (2,))
    win.mode_combo.setCurrentIndex(6)
    assert win.build_config().mode == "unico"


def test_eventos_por_fase_reinician_las_barras_y_muestran_la_fase():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.handle_event({"type": "start", "fase": 1, "total": 4, "workers": 1, "batches": 1})
    assert "1" in win.phase_label.text() and win.overall.maximum() == 4
    win.handle_event({"type": "image_done", "worker": 0, "name": "a.jpg", "status": "cropped", "reason": "",
                      "head": 0.0, "torso": 0.0, "area_ratio": 0.8, "fase": 1})
    win.handle_event({"type": "phase_done", "fase": 1, "counts": {"cropped": 1}, "cancelled": False})
    win.handle_event({"type": "start", "fase": 2, "total": 3, "workers": 2, "batches": 1})
    assert "2" in win.phase_label.text() and win.overall.maximum() == 3 and win.overall.value() == 0
    assert len(win.worker_bars) == 2


def test_error_fatal_del_runner_reactiva_el_boton_iniciar(tmp_path):
    from acfc.core.runner import RunConfig
    from acfc.gui import RunnerThread
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    got = []
    th = RunnerThread(RunConfig(tmp_path, mode="fases", phases=(2,), backend="fake"))   # falta fase1_mapa
    th.event.connect(got.append)
    th.run()
    assert got and got[-1]["type"] == "fatal" and "falta la carpeta" in got[-1]["error"]
    win.start_btn.setEnabled(False)
    win.handle_event(got[-1])
    assert win.start_btn.isEnabled() and "falta la carpeta" in win.status.text()
