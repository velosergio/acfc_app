from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from acfc.core.runner import RunConfig, make_runner
from acfc.core.timing import eta_seconds, fmt_hms


class RunnerThread(QThread):
    event = Signal(dict)

    def __init__(self, cfg: RunConfig):
        super().__init__()
        self.runner = make_runner(cfg)

    def run(self) -> None:
        try:
            for ev in self.runner.events():
                self.event.emit(ev)
        except Exception as exc:  # p. ej. falta la carpeta de la fase anterior
            self.event.emit({"type": "fatal", "error": str(exc)})

    def cancel(self) -> None:
        self.runner.cancel()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ACFC – Procesador de imágenes")
        self.resize(760, 640)
        self.thread_: RunnerThread | None = None
        self.worker_bars: list[QProgressBar] = []

        self.path_edit = QLineEdit()
        browse = QPushButton("Examinar…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.path_edit)
        row.addWidget(browse)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 8)
        self.workers_spin.setValue(2)
        self.force_check = QCheckBox("Rehacer las que ya existen en la carpeta de salida")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Por fases (1→2→3→4)", "Fases 1, 3 y 4 (sin personas)", "Solo fase 1 (mapa)",
                                  "Solo fase 2 (personas)", "Solo fase 3 (texto)", "Solo fase 4 (tamaño mínimo)",
                                  "Una sola pasada"])
        self.min_side_spin = QSpinBox()
        self.min_side_spin.setRange(0, 10000)
        self.min_side_spin.setValue(500)
        self.min_side_spin.setSuffix(" px")
        form = QFormLayout()
        form.addRow("Carpeta:", row)
        form.addRow("Modo:", self.mode_combo)
        form.addRow("Workers:", self.workers_spin)
        form.addRow("Lado mínimo (fase 4):", self.min_side_spin)
        form.addRow("", self.force_check)

        self.start_btn = QPushButton("Iniciar")
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn.clicked.connect(self._cancel)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.cancel_btn)

        self.overall = QProgressBar()
        self.overall.setFormat("General: %v / %m")
        self.phase_label = QLabel("")
        self.time_label = QLabel("Tiempo: --:--:--")
        self.workers_box = QGroupBox("Workers")
        self.workers_layout = QVBoxLayout(self.workers_box)
        self.status = QLabel("Listo.")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Archivo", "Motivo", "Gorra", "Torso"])
        self.table.horizontalHeader().setStretchLastSection(True)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addLayout(form)
        lay.addLayout(buttons)
        lay.addWidget(self.overall)
        lay.addWidget(self.phase_label)
        lay.addWidget(self.time_label)
        lay.addWidget(self.workers_box)
        lay.addWidget(self.status)
        lay.addWidget(QLabel("Imágenes descartadas:"))
        lay.addWidget(self.table)
        self.setCentralWidget(central)

        self._now = time.monotonic
        self._t0: float | None = None          # inicio de toda la corrida
        self._phase_t0: float | None = None    # inicio de la fase actual
        self._phase_done = 0
        self._phase_total = 0
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh_time)

    def begin_timing(self) -> None:
        self._t0 = self._phase_t0 = self._now()
        self._phase_done = self._phase_total = 0
        self.refresh_time()

    def refresh_time(self) -> None:
        if self._t0 is None:
            return
        now = self._now()
        txt = f"Tiempo total {fmt_hms(now - self._t0)}"
        if self._phase_t0 is not None and self._phase_total:
            eta = eta_seconds(now - self._phase_t0, self._phase_done, self._phase_total)
            txt += f" · ETA fase {fmt_hms(eta)}"
        self.time_label.setText(txt)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Carpeta de imágenes")
        if d:
            self.path_edit.setText(d)

    def build_config(self) -> RunConfig:
        idx = self.mode_combo.currentIndex()
        phases = {0: (1, 2, 3, 4), 1: (1, 3, 4), 2: (1,), 3: (2,), 4: (3,), 5: (4,)}.get(idx, (1, 2, 3, 4))
        return RunConfig(Path(self.path_edit.text().strip()), workers=self.workers_spin.value(),
                         force=self.force_check.isChecked(), mode="unico" if idx == 6 else "fases", phases=phases,
                         skip_people=idx == 1, min_side=self.min_side_spin.value())

    def _start(self) -> None:
        folder = Path(self.path_edit.text().strip())
        if not folder.is_dir():
            QMessageBox.warning(self, "Carpeta", "La carpeta no existe.")
            return
        self.table.setRowCount(0)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status.setText("Procesando…")
        self.begin_timing()
        self._timer.start()
        self.thread_ = RunnerThread(self.build_config())
        self.thread_.event.connect(self.handle_event)
        self.thread_.start()

    def _cancel(self) -> None:
        if self.thread_:
            self.thread_.cancel()
            self.status.setText("Cancelando…")

    def handle_event(self, ev: dict) -> None:
        t = ev["type"]
        if t == "start":
            self.overall.setRange(0, ev["total"])
            self.overall.setValue(0)
            self._phase_t0, self._phase_done, self._phase_total = self._now(), 0, ev["total"]
            if self._t0 is None:                 # eventos sin pasar por _start (p. ej. pruebas)
                self._t0 = self._phase_t0
            self.phase_label.setText(f"Fase {ev['fase']}" if ev.get("fase") else "")
            while self.workers_layout.count():
                self.workers_layout.takeAt(0).widget().deleteLater()
            self.worker_bars = []
            for w in range(ev["workers"]):
                bar = QProgressBar()
                bar.setFormat(f"Worker {w}: %v / %m")
                self.workers_layout.addWidget(bar)
                self.worker_bars.append(bar)
        elif t == "batch_start":
            bar = self.worker_bars[ev["worker"]]
            bar.setRange(0, ev["size"])
            bar.setValue(0)
            bar.setFormat(f"Worker {ev['worker']} lote {ev['batch']}: %v / %m")
        elif t == "image_done":
            bar = self.worker_bars[ev["worker"]]
            bar.setValue(bar.value() + 1)
            self.overall.setValue(self.overall.value() + 1)
            self._phase_done += 1
            if ev["status"] == "discarded":
                r = self.table.rowCount()
                self.table.insertRow(r)
                for c, v in enumerate([ev["name"], ev["reason"], f'{ev["head"]:.2f}', f'{ev["torso"]:.2f}']):
                    self.table.setItem(r, c, QTableWidgetItem(v))
        elif t == "worker_error":
            self.status.setText(f"Error en worker {ev['worker']}: {ev['error']}")
        elif t == "phase_done":
            pass
        elif t == "fatal":
            self.status.setText(f"Error: {ev['error']}")
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
        elif t == "finished":
            c = ev["counts"]
            estado = "Cancelado" if ev["cancelled"] else "Terminado"
            if ev.get("phases"):
                detalle = " | ".join(f"Fase {n}: " + " ".join(f"{k}={v}" for k, v in sorted(cn.items()))
                                     for n, cn in sorted(ev["phases"].items()))
                self.status.setText(f"{estado}. {detalle}")
            else:
                self.status.setText(f"{estado}. guardadas={c.get('saved', 0)} revisar={c.get('review', 0)} "
                                    f"descartadas={c.get('discarded', 0)} errores={c.get('error', 0)}")
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self._timer.stop()
            self.refresh_time()

    def closeEvent(self, e) -> None:  # noqa: N802
        if self.thread_ and self.thread_.isRunning():
            self.thread_.cancel()
            self.thread_.wait(10000)
        super().closeEvent(e)


def run_gui() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()
