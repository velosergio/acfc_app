# ACFC Modo por fases – Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Añadir un modo por fases (1: recorte por mapa, 2: descarte de personas de chaleco y gorra blancos, 3: limpieza del texto) con carpeta de salida por fase, sin eliminar el modo de una sola pasada.

**Architecture:** Tres funciones por imagen en `acfc/core/phases.py` que reutilizan los detectores existentes. `Runner` se generaliza con `cfg.phase` (0 = una pasada, 1/2/3 = fases) y un `PhasedRunner` encadena las fases emitiendo los mismos eventos más `phase_start/phase_done`. La fase 1 deja `manifest.csv` con el lado del corte, que la fase 3 usa para leer solo la franja junto a ese borde.

**Tech Stack:** Python 3.12 (venv ya creado en `acfc_app/.venv`), PIL, numpy, OpenCV, ultralytics, easyocr, PySide6, rich, pytest.

**Spec:** `docs/superpowers/specs/2026-10-08-acfc-modo-por-fases-design.md` (extiende `2026-10-07-acfc-procesador-imagenes-design.md`).

## Global Constraints

- No eliminar ni romper nada existente: `process_image`, `Runner` en modo 0, `--modo unico`, los 52 tests actuales siguen en verde.
- Carpetas bajo `<ruta>/`: `fase1_mapa/`, `fase2_personas/`, `outputs/` (final = fase 3). Cada fase lee la carpeta de la anterior; la fase 1 lee `<ruta>`.
- Fase 1: sin GPU ni modelos; si hay minimapa corta la franja de todo el ancho a la altura del mapa (margen `int(OverlayParams.margin * max(W, H))`) desde su borde y guarda JPEG calidad 95 (escritura atómica); si no hay, copia el archivo sin recomprimir. `manifest.csv`: `archivo,lado,corte` con `lado` ∈ `arriba|abajo|ninguno`.
- Fase 2: solo YOLO; la regla vigente (gorra blanca Y uniforme blanco → descartar); descartadas no se copian y van a `log_fase2_descartadas.csv`; las demás se copian sin recomprimir y su fila del manifiesto pasa a `fase2_personas/manifest.csv`.
- Fase 3: solo EasyOCR; OCR únicamente en la franja del lado indicado por el manifiesto (ambas franjas si `ninguno` o si falta); quita el bloque de texto con `largest_free_rect`; sin detección de brújula; área mínima 40 % → `log_fase3_revisar.csv`.
- Lotes de 5; reanudable (omite lo ya existente en la carpeta de salida de la fase salvo `--force`; `--force` reinicia logs y manifiesto de esa fase); escritura atómica; ninguna imagen detiene la corrida.
- Barra de progreso por worker y general (por fase) en CLI y GUI; la GUI añade selector de modo/fase.
- Todo el texto al usuario en español.
- Los umbrales de escala del detector se miden contra el **lado corto** (`min(W, H)`), no contra el ancho.

## Review Focus

- Minimapa arriba (no abajo) → se corta lo de arriba y se conserva lo de abajo (Task 3).
- Corte que deja menos del área mínima → se guarda y se anota en `log_fase1_revisar.csv` (Task 3).
- Foto apaisada o con orientación EXIF (el lado largo no es la altura) → coordenadas sobre la imagen ya transpuesta (Task 1, 5).
- Fase 3 lanzada sola sin manifiesto → lee las dos franjas (Task 5, 6).
- Lanzar una fase cuya carpeta de entrada no existe → error claro, no traza (Task 6, 8).
- Carpeta de entrada vacía en una fase → termina limpio (Task 6).
- Cancelar entre fases o dentro de una → no sigue con la siguiente ni deja `.tmp` (Task 7).
- Rehacer (`--force`) no deja filas duplicadas en logs ni manifiestos (Task 6).

---

### Task 1: Detector de minimapa – escala por lado corto y pines pequeños

**Files:**
- Modify: `acfc/core/overlay.py` (`OverlayParams`, `find_minimap`, `find_compass`, `text_chain`)
- Test: `tests/test_overlay.py`

**Interfaces:**
- Consumes: nada nuevo.
- Produces: `find_minimap(rgb, p)` sin cambio de firma; ahora acepta fotos apaisadas y pines pequeños. `OverlayParams.pin_area_min` pasa a `0.00012`.

Causas medidas en fotos reales: `8028a0bab8273c91` (1600×1200) tiene el pin en 844 px² frente a un mínimo de 1024 (calculado con el ancho, que en apaisada es el lado largo); `cc6ae1bd9882ca07` (899×1599) tiene un pin de ~200 px² frente a un mínimo de 323.

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_overlay.py`:

```python
def _tile_pin_chico(img, x0, y0, x1, y1, radio=4):
    """Mosaico con un pin pequeño (radio de cabeza 4 px)."""
    img[y0:y1, x0:x1] = (197, 232, 197)
    cx, tip = (x0 + x1) // 2, (y0 + y1) // 2
    cv2.circle(img, (cx, tip - 2 * radio), radio, (234, 67, 53), -1)
    cv2.fillPoly(img, [np.array([[cx - radio + 1, tip - radio - 1], [cx + radio - 1, tip - radio - 1], [cx, tip]], np.int32)], (234, 67, 53))


def test_minimapa_en_foto_apaisada_usa_el_lado_corto():
    img = _fondo(800, 500)                            # apaisada: el lado corto es 500
    _tile_pin_chico(img, 0, 350, 150, 500, radio=5)   # tile 150x150 abajo a la izquierda
    (x0, y0, x1, y1), = find_minimap(img, P)
    assert x0 == 0 and y1 == 500 and abs(x1 - 150) <= 6 and abs(y0 - 350) <= 6


def test_minimapa_con_pin_pequeno_en_foto_vertical():
    img = _fondo(900, 1600)
    _tile_pin_chico(img, 0, 0, 250, 250, radio=5)
    (x0, y0, x1, y1), = find_minimap(img, P)
    assert (x0, y0) == (0, 0) and abs(x1 - 250) <= 8 and abs(y1 - 250) <= 8
```

- [ ] **Step 2: Verificar que fallan**

Run: `.\.venv\Scripts\python -m pytest tests/test_overlay.py -q`
Expected: los dos tests nuevos FAIL (`ValueError: not enough values to unpack`, no se detecta el minimapa).

- [ ] **Step 3: Implementar**

En `acfc/core/overlay.py`:

1. `pin_area_min: float = 0.00012` (era `0.0004`).
2. En `find_minimap`, definir `s = min(H, W)` y reemplazar los usos de `W` como escala:
   - `region_w, region_h = int(p.tile_max_w * s / 2) + 2, min(H, int(0.4 * s))`
   - `p.pin_area_min * s * s <= area <= p.pin_area_max * s * s`
   - `p.tile_min_w * s <= tw <= p.tile_max_w * s`
   Mantener `W` solo para posicionar (`W - tw`).
3. En `find_compass`: `s = min(H, W)`; `size = int(p.compass_size * s)`; `area >= p.needle_area_min * s * s`.
4. En `text_chain`: `s = min(width, height)`; la condición de altura de línea pasa a `p.line_h_min * s <= alto <= p.line_h_max * s`.

- [ ] **Step 4: Verificar que pasan**

Run: `.\.venv\Scripts\python -m pytest -q`
Expected: toda la suite en verde (los tests existentes usan imágenes verticales donde `min(W,H) == W`).

- [ ] **Step 5: Medir el efecto sobre las 1932 fotos y revisar**

Crear `tools/map_stats.py`:
```python
import json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageOps

from acfc.core.overlay import OverlayParams, find_minimap
from acfc.core.runner import list_images


def main() -> None:
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows, c = {}, Counter()
    for f in list_images(folder):
        img = np.asarray(ImageOps.exif_transpose(Image.open(f)).convert("RGB"))
        maps = find_minimap(img, OverlayParams())
        rows[f.name] = {"W": img.shape[1], "H": img.shape[0], "maps": maps}
        c["sin minimapa" if not maps else ("arriba" if maps[0][1] == 0 else "abajo")] += 1
    out.write_text(json.dumps(rows))
    print(dict(c), "total", len(rows))
    for n in ("8028a0bab8273c91.jpeg", "cc6ae1bd9882ca07.jpeg"):
        print(n, "detectado" if rows.get(n, {}).get("maps") else "SIN DETECTAR")


if __name__ == "__main__":
    main()
```
Ejecutar:
```powershell
.\.venv\Scripts\python tools\map_stats.py "C:\Users\sergi\Documents\Nando\ACFC\ACFC" "$env:TEMP\maps.json"
```
Resultado previo: 1264 abajo, 36 arriba, 632 sin detección. Esperado: las dos fotos conocidas con `detectado` y el total sin detección más bajo. Luego generar una hoja con 15 fotos al azar de las que siguen sin detección (las claves de `maps.json` con `maps` vacío) y comprobar a ojo cuántas siguen teniendo mapa; si quedan fotos con mapa sin detectar en esa muestra, diagnosticar con el mismo método (componentes rojos, área, relación alto/ancho, color) y repetir. Si `cc6ae1bd9882ca07` sigue sin detectarse porque el pin sale partido en dos componentes, aplicar un cierre morfológico 3×3 a la máscara roja antes de `_components` y añadir un test con un pin partido por un hueco de 1 px.

- [ ] **Step 6: Commit**

```powershell
git add acfc/core/overlay.py tests/test_overlay.py tools/map_stats.py
git commit -m "fix: minimapa escala por lado corto y acepta pines pequenos"
```

---

### Task 2: Manifiesto y campos de resultado por fase

**Files:**
- Create: `acfc/core/manifest.py`
- Modify: `acfc/core/pipeline.py` (`ImageResult`)
- Test: `tests/test_manifest.py`

**Interfaces:**
- Produces:
  - `MANIFEST = "manifest.csv"`; `SIDES = ("arriba", "abajo", "ninguno")`
  - `write_row(folder: Path, name: str, side: str, cut: int) -> None` (agrega al CSV; crea con cabecera `archivo,lado,corte`)
  - `read_manifest(folder: Path) -> dict[str, tuple[str, int]]` (vacío si no existe)
  - `ImageResult` gana `side: str = ""` y `cut: int = 0` (al final, con valores por defecto).

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_manifest.py`
```python
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
```

- [ ] **Step 2: Verificar que fallan**

Run: `.\.venv\Scripts\python -m pytest tests/test_manifest.py -q`
Expected: FAIL (`ModuleNotFoundError: acfc.core.manifest`).

- [ ] **Step 3: Implementar**

`acfc/core/manifest.py`
```python
from __future__ import annotations

import csv
from pathlib import Path

MANIFEST = "manifest.csv"
SIDES = ("arriba", "abajo", "ninguno")


def write_row(folder: Path, name: str, side: str, cut: int) -> None:
    path = Path(folder) / MANIFEST
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["archivo", "lado", "corte"])
        w.writerow([name, side, int(cut)])


def read_manifest(folder: Path) -> dict[str, tuple[str, int]]:
    path = Path(folder) / MANIFEST
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["archivo"]: (r["lado"], int(r["corte"])) for r in csv.DictReader(fh)}
```

En `acfc/core/pipeline.py`, agregar al final de `ImageResult`:
```python
    side: str = ""
    cut: int = 0
```

- [ ] **Step 4: Verificar que pasan**

Run: `.\.venv\Scripts\python -m pytest -q`
Expected: todo en verde.

- [ ] **Step 5: Commit**

```powershell
git add acfc/core/manifest.py acfc/core/pipeline.py tests/test_manifest.py
git commit -m "feat: manifiesto de fases y campos lado/corte en ImageResult"
```

---

### Task 3: Fase 1 – recorte por mapa

**Files:**
- Create: `acfc/core/phases.py`
- Test: `tests/test_phases.py`

**Interfaces:**
- Consumes: `find_minimap`, `OverlayParams` (Task 1); `ImageResult` con `side/cut` (Task 2); `_save_atomic` de `acfc.core.pipeline`.
- Produces:
  - `copy_atomic(src: Path, dest: Path) -> None` (copia byte a byte vía temporal + `os.replace`)
  - `phase1_map(path: Path, out_dir: Path, params=OverlayParams(), min_keep: float = 0.4, force: bool = False) -> ImageResult` con `status` ∈ `cropped | copied | review | skipped | error`; `side` ∈ `arriba|abajo|ninguno`; `cut` = filas retiradas (borde de corte). Nunca lanza excepciones.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_phases.py`
```python
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
```

- [ ] **Step 2: Verificar que fallan**

Run: `.\.venv\Scripts\python -m pytest tests/test_phases.py -q`
Expected: FAIL (`ModuleNotFoundError: acfc.core.phases`).

- [ ] **Step 3: Implementar**

`acfc/core/phases.py`
```python
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from acfc.core.overlay import OverlayParams, find_minimap
from acfc.core.pipeline import ImageResult, _save_atomic


def copy_atomic(src: Path, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def phase1_map(path: Path, out_dir: Path, params: OverlayParams = OverlayParams(),
               min_keep: float = 0.4, force: bool = False) -> ImageResult:
    dest = out_dir / path.name
    try:
        if dest.exists() and not force:
            return ImageResult(path.name, "skipped", "ya existe")
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw).convert("RGB")
        w, h = img.size
        maps = find_minimap(np.asarray(img), params)
        if not maps:
            copy_atomic(path, dest)
            return ImageResult(path.name, "copied", "sin minimapa", side="ninguno", cut=0)
        x0, y0, x1, y1 = max(maps, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        margin = int(params.margin * max(w, h))
        if y0 == 0:                                   # mapa arriba: se retira lo de arriba
            side, cut = "arriba", min(h, y1 + margin)
            rect = (0, cut, w, h)
        else:                                         # mapa abajo: se retira lo de abajo
            side, cut = "abajo", max(0, y0 - margin)
            rect = (0, 0, w, cut)
        keep = (rect[3] - rect[1]) / h
        _save_atomic(img.crop(rect), dest)
        status = "review" if keep < min_keep else "cropped"
        reason = f"corte deja {keep:.0%}" if status == "review" else ""
        return ImageResult(path.name, status, reason, area_ratio=keep, side=side, cut=cut)
    except Exception as exc:  # una imagen mala nunca detiene la corrida
        return ImageResult(path.name, "error", f"{type(exc).__name__}: {exc}")
```

- [ ] **Step 4: Verificar que pasan**

Run: `.\.venv\Scripts\python -m pytest tests/test_phases.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```powershell
git add acfc/core/phases.py tests/test_phases.py
git commit -m "feat: fase 1 recorte por mapa"
```

---

### Task 4: Fase 2 – descarte de personas

**Files:**
- Modify: `acfc/core/phases.py`
- Test: `tests/test_phases.py`

**Interfaces:**
- Consumes: `copy_atomic` (Task 3); `build_backend("fake")` (existente: `FakePeople` descarta si el píxel (0,0) es rojo).
- Produces: `phase2_people(path: Path, out_dir: Path, people, force: bool = False) -> ImageResult`; `status` ∈ `copied | discarded | skipped | error`; las copiadas llevan `head/torso`.

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_phases.py`:
```python
from acfc.core.backends import build_backend
from acfc.core.phases import phase2_people


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
```

- [ ] **Step 2: Verificar que fallan**

Run: `.\.venv\Scripts\python -m pytest tests/test_phases.py -q`
Expected: FAIL (`ImportError: cannot import name 'phase2_people'`).

- [ ] **Step 3: Implementar**

Añadir a `acfc/core/phases.py`:
```python
def phase2_people(path: Path, out_dir: Path, people, force: bool = False) -> ImageResult:
    dest = out_dir / path.name
    try:
        if dest.exists() and not force:
            return ImageResult(path.name, "skipped", "ya existe")
        with Image.open(path) as raw:
            rgb = np.asarray(ImageOps.exif_transpose(raw).convert("RGB"))
        d = people.check(rgb)
        if d.discard:
            return ImageResult(path.name, "discarded", "uniforme y gorra blancos", d.head, d.torso)
        copy_atomic(path, dest)
        return ImageResult(path.name, "copied", "", d.head, d.torso)
    except Exception as exc:
        return ImageResult(path.name, "error", f"{type(exc).__name__}: {exc}")
```

- [ ] **Step 4: Verificar que pasan**

Run: `.\.venv\Scripts\python -m pytest tests/test_phases.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```powershell
git add acfc/core/phases.py tests/test_phases.py
git commit -m "feat: fase 2 descarte de personas"
```

---

### Task 5: Fase 3 – limpieza del texto

**Files:**
- Modify: `acfc/core/overlay.py` (`OverlayDetector.text_boxes`, `detect` lo reutiliza), `acfc/core/backends.py` (`FakeOverlay.text_boxes`), `acfc/core/phases.py`
- Test: `tests/test_phases.py`

**Interfaces:**
- Consumes: `text_chain`, `pad_box` (existentes); `largest_free_rect`; `copy_atomic`, `_save_atomic`.
- Produces:
  - `OverlayDetector.text_boxes(rgb, sides: tuple[str, ...]) -> list[Box]` (OCR solo en las franjas de `sides`, cajas ya con margen; usa `batch_size=8`, que midió resultados idénticos y ~24 % menos tiempo)
  - `FakeOverlay.text_boxes(rgb, sides)`: guarda `self.last_sides = sides`; devuelve `(0, 0, w//5, h//10)` si `"arriba"` está en `sides` y `(w - w//5, h - h//10, w, h)` si `"abajo"`.
  - `phase3_text(path: Path, out_dir: Path, overlay, side: str = "ninguno", min_area_ratio: float = 0.4, force: bool = False) -> ImageResult`; `status` ∈ `saved | review | skipped | error`. `sides = (side,)` si `side` ∈ `arriba|abajo`, si no `("arriba", "abajo")`.

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_phases.py`:
```python
from acfc.core.phases import phase3_text


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
```

- [ ] **Step 2: Verificar que fallan**

Run: `.\.venv\Scripts\python -m pytest tests/test_phases.py -q`
Expected: FAIL (`ImportError: cannot import name 'phase3_text'`).

- [ ] **Step 3: Implementar**

En `acfc/core/backends.py`, reemplazar `FakeOverlay` por:
```python
class FakeOverlay:
    last_sides: tuple = ()

    def detect(self, rgb) -> list[Box]:
        h, w = rgb.shape[:2]
        return [(0, 0, w // 5, h // 10)]

    def text_boxes(self, rgb, sides) -> list[Box]:
        self.last_sides = tuple(sides)
        h, w = rgb.shape[:2]
        boxes = []
        if "arriba" in sides:
            boxes.append((0, 0, w // 5, h // 10))
        if "abajo" in sides:
            boxes.append((w - w // 5, h - h // 10, w, h))
        return boxes
```

En `acfc/core/overlay.py`, dentro de `OverlayDetector`, agregar `text_boxes` y hacer que `detect` lo use:
```python
    def text_boxes(self, rgb: np.ndarray, sides: tuple[str, ...]) -> list[Box]:
        H, W = rgb.shape[:2]
        strip = int(self.p.strip_ratio * H)
        candidates = []
        for y0, at_top, lado in ((0, True, "arriba"), (H - strip, False, "abajo")):
            if lado not in sides:
                continue
            results = self.reader.readtext(np.ascontiguousarray(rgb[y0:y0 + strip]), low_text=0.3,
                                           text_threshold=0.5, contrast_ths=0.3, adjust_contrast=0.7, batch_size=8)
            found = text_chain(results, W, H, y0, at_top, self.p)
            if found:
                candidates.append(found)
        if not candidates:
            return []
        return [pad_box(max(candidates, key=lambda c: c[1])[0], W, H, self.p.margin)]  # un solo bloque por foto

    def detect(self, rgb: np.ndarray) -> list[Box]:
        H, W = rgb.shape[:2]
        boxes = self.text_boxes(rgb, ("arriba", "abajo"))
        boxes += [pad_box(b, W, H, self.p.margin) for b in find_minimap(rgb, self.p) + find_compass(rgb, self.p)]
        return boxes
```
(Eliminar el cuerpo anterior de `detect`.)

Añadir a `acfc/core/phases.py`:
```python
from acfc.core.crop import largest_free_rect


def phase3_text(path: Path, out_dir: Path, overlay, side: str = "ninguno",
                min_area_ratio: float = 0.4, force: bool = False) -> ImageResult:
    dest = out_dir / path.name
    try:
        if dest.exists() and not force:
            return ImageResult(path.name, "skipped", "ya existe")
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw).convert("RGB")
        w, h = img.size
        sides = (side,) if side in ("arriba", "abajo") else ("arriba", "abajo")
        boxes = overlay.text_boxes(np.asarray(img), sides)
        if not boxes:
            copy_atomic(path, dest)
            return ImageResult(path.name, "saved", "sin texto", area_ratio=1.0)
        rect = largest_free_rect(w, h, boxes)
        if rect is None:
            _save_atomic(img, dest)
            return ImageResult(path.name, "review", "sin rectangulo libre; se guardo el original", area_ratio=1.0)
        ratio = (rect[2] - rect[0]) * (rect[3] - rect[1]) / (w * h)
        _save_atomic(img.crop(rect), dest)
        if ratio < min_area_ratio:
            return ImageResult(path.name, "review", f"recorte pequeno ({ratio:.0%})", area_ratio=ratio)
        return ImageResult(path.name, "saved", "", area_ratio=ratio)
    except Exception as exc:
        return ImageResult(path.name, "error", f"{type(exc).__name__}: {exc}")
```

- [ ] **Step 4: Verificar que pasan**

Run: `.\.venv\Scripts\python -m pytest -q`
Expected: todo en verde (los tests de `detect` con backend falso y overlay siguen pasando).

- [ ] **Step 5: Commit**

```powershell
git add acfc/core/overlay.py acfc/core/backends.py acfc/core/phases.py tests/test_phases.py
git commit -m "feat: fase 3 limpieza de texto con OCR solo en la franja necesaria"
```

---

### Task 6: Runner por fase (config, carpetas, manifiestos, logs, force)

**Files:**
- Modify: `acfc/core/runner.py`, `acfc/core/backends.py`
- Test: `tests/test_runner_phases.py`

**Interfaces:**
- Consumes: `phase1_map`, `phase2_people`, `phase3_text` (Tasks 3–5); `write_row`, `read_manifest` (Task 2).
- Produces:
  - `RunConfig` gana `phase: int = 0`, `mode: str = "unico"`, `phases: tuple[int, ...] = (1, 2, 3)`, `sides: dict[str, str] = field(default_factory=dict)`; propiedades `in_dir` y `out_dir` según `phase` (0: `input/→input/outputs`, 1: `input/→fase1_mapa`, 2: `fase1_mapa→fase2_personas`, 3: `fase2_personas→outputs`).
  - `build_backend(name: str, phase: int = 0)` → `(people, overlay)`; con `"real"` carga solo lo necesario (fase 1: `(None, None)`; fase 2: `(PersonFilter(), None)`; fase 3: `(None, OverlayDetector())`; fase 0: ambos). `"fake"` sigue devolviendo ambos falsos.
  - `Runner(cfg).events()` funciona para `cfg.phase` 0..3; el evento `start` y `finished` llevan `fase`; si `cfg.in_dir` no existe lanza `FileNotFoundError` con el mensaje `falta la carpeta de la fase anterior: <ruta>`.
  - Logs por fase en `cfg.out_dir`: fase 0 como hasta ahora; fases 1–3: `log_fase{n}_descartadas.csv`, `log_fase{n}_errores.csv`, `log_fase{n}_revisar.csv`.
  - Fases 1 y 2 escriben `manifest.csv` en su `out_dir` desde el proceso padre (un solo escritor). `--force` (`cfg.force`) borra logs y manifiesto de esa fase al empezar.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_runner_phases.py`
```python
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from acfc.core.manifest import read_manifest
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
    make_image(f1 / "ok.jpg"); make_image(f1 / "blanca.jpg", color=(255, 0, 0))
    from acfc.core.manifest import write_row
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
    make_image(f2 / "a.jpg", size=(100, 100)); make_image(f2 / "b.jpg", size=(100, 100))
    from acfc.core.manifest import write_row
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
    make_image(f1 / "blanca.jpg", color=(255, 0, 0))
    from acfc.core.manifest import write_row
    write_row(f1, "blanca.jpg", "ninguno", 0)
    _run(RunConfig(tmp_path, phase=2, workers=1, backend="fake"))
    _run(RunConfig(tmp_path, phase=2, workers=1, backend="fake", force=True))
    out = tmp_path / "fase2_personas"
    desc = list(csv.DictReader((out / "log_fase2_descartadas.csv").open(encoding="utf-8")))
    assert len(desc) == 1                                     # sin fila duplicada de la corrida anterior
```

- [ ] **Step 2: Verificar que fallan**

Run: `.\.venv\Scripts\python -m pytest tests/test_runner_phases.py -q`
Expected: FAIL (`TypeError: RunConfig.__init__() got an unexpected keyword argument 'phase'`).

- [ ] **Step 3: Implementar**

`acfc/core/backends.py`: reemplazar `build_backend` por:
```python
def build_backend(name: str, phase: int = 0):
    if name == "fake":
        return FakePeople(), FakeOverlay()
    people = overlay = None
    if phase in (0, 2):
        from acfc.core.people import PersonFilter
        people = PersonFilter()
    if phase in (0, 3):
        from acfc.core.overlay import OverlayDetector
        overlay = OverlayDetector()
    return people, overlay
```

`acfc/core/runner.py` (cambios):
```python
from dataclasses import asdict, dataclass, field, replace

from acfc.core.manifest import read_manifest, write_row
from acfc.core.phases import phase1_map, phase2_people, phase3_text
```
- `RunConfig`: agregar los campos `phase`, `mode`, `phases`, `sides` (con `field(default_factory=dict)`) y reemplazar la propiedad `out_dir` por:
```python
    @property
    def in_dir(self) -> Path:
        base = Path(self.input_dir)
        return {0: base, 1: base, 2: base / "fase1_mapa", 3: base / "fase2_personas"}[self.phase]

    @property
    def out_dir(self) -> Path:
        base = Path(self.input_dir)
        return {0: base / "outputs", 1: base / "fase1_mapa", 2: base / "fase2_personas", 3: base / "outputs"}[self.phase]
```
- Función de despacho y `worker_main`:
```python
def _process(cfg: RunConfig, path: Path, people, overlay):
    if cfg.phase == 1:
        return phase1_map(path, cfg.out_dir, min_keep=cfg.min_area_ratio, force=cfg.force)
    if cfg.phase == 2:
        return phase2_people(path, cfg.out_dir, people, cfg.force)
    if cfg.phase == 3:
        return phase3_text(path, cfg.out_dir, overlay, cfg.sides.get(path.name, "ninguno"), cfg.min_area_ratio, cfg.force)
    return process_image(path, cfg.out_dir, people, overlay, cfg.min_area_ratio, cfg.force)
```
En `worker_main`: `people, overlay = build_backend(cfg.backend, cfg.phase)` y `res = _process(cfg, Path(p), people, overlay)`.
- `_Logs(out_dir, phase)`: nombres según la fase:
```python
    BASE = {"discarded": ("descartadas", ["archivo", "motivo", "score_gorra", "score_torso"]),
            "error": ("errores", ["archivo", "error"]),
            "review": ("revisar", ["archivo", "area_ratio", "motivo"])}

    def __init__(self, out_dir: Path, phase: int = 0):
        self.out_dir, self.phase = out_dir, phase

    def filename(self, kind: str) -> str:
        suffix = self.BASE[kind][0]
        return f"log_{suffix}.csv" if self.phase == 0 else f"log_fase{self.phase}_{suffix}.csv"
```
y en `write`, usar `self.BASE[ev["status"]]` para el encabezado y `self.filename(...)` para la ruta (mantener el mapa estado→tipo: `discarded`, `error`, `review`). Para fase 0 los nombres quedan `log_descartadas.csv`, `log_errores.csv`, `log_revisar.csv` como antes.
- `Runner.events()`:
  1. `cfg = self.cfg`; si `not cfg.in_dir.is_dir()`: `raise FileNotFoundError(f"falta la carpeta de la fase anterior: {cfg.in_dir}")`.
  2. `cfg.out_dir.mkdir(...)`; si `cfg.force`: borrar los tres logs de `_Logs(cfg.out_dir, cfg.phase)` y, si `cfg.phase in (1, 2)`, `manifest.csv`.
  3. `files = list_images(cfg.in_dir)`; si `cfg.phase == 3`: `cfg = replace(cfg, sides={n: s for n, (s, _c) in read_manifest(cfg.in_dir).items()})`; si `cfg.phase == 2`: `prev = read_manifest(cfg.in_dir)`.
  4. Eventos `start` y `finished` incluyen `"fase": cfg.phase`.
  5. Al recibir `image_done`: contar, `logs.write(ev)`, y además: fase 1 → `if ev["status"] in ("cropped", "copied", "review"): write_row(cfg.out_dir, ev["name"], ev["side"], ev["cut"])`; fase 2 → `if ev["status"] == "copied": side, cut = prev.get(ev["name"], ("ninguno", 0)); write_row(cfg.out_dir, ev["name"], side, cut)`.
  - El `Process(... args=(i, task_q, event_q, cfg, ...))` debe usar la `cfg` ya reemplazada (con `sides`).
  - Usar `_Logs(cfg.out_dir, cfg.phase)`.

- [ ] **Step 4: Verificar que pasan**

Run: `.\.venv\Scripts\python -m pytest -q`
Expected: todo en verde, incluidos `tests/test_runner.py` del modo 0.

- [ ] **Step 5: Commit**

```powershell
git add acfc/core/runner.py acfc/core/backends.py tests/test_runner_phases.py
git commit -m "feat: runner por fases con manifiestos, logs por fase y force"
```

---

### Task 7: PhasedRunner (encadenar fases) y `make_runner`

**Files:**
- Modify: `acfc/core/runner.py`
- Test: `tests/test_phased_runner.py`

**Interfaces:**
- Consumes: `Runner`, `RunConfig` (Task 6).
- Produces:
  - `PhasedRunner(cfg: RunConfig)` con `.events()` y `.cancel()`; recorre `cfg.phases`; por cada fase emite los eventos del `Runner` interno (cada uno con `fase`), convierte el `finished` interno en `{"type": "phase_done", "fase": n, "counts": {...}, "cancelled": bool}` y al final emite `{"type": "finished", "counts": <counts de la última fase ejecutada>, "phases": {n: counts}, "cancelled": bool}`.
  - `make_runner(cfg: RunConfig)` → `PhasedRunner(cfg)` si `cfg.mode == "fases"`, si no `Runner(cfg)`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_phased_runner.py`
```python
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
```

- [ ] **Step 2: Verificar que fallan**

Run: `.\.venv\Scripts\python -m pytest tests/test_phased_runner.py -q`
Expected: FAIL (`ImportError: cannot import name 'PhasedRunner'`).

- [ ] **Step 3: Implementar**

Añadir al final de `acfc/core/runner.py`:
```python
class PhasedRunner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self._current: Runner | None = None
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        if self._current:
            self._current.cancel()

    def events(self) -> Iterator[dict]:
        per_phase: dict[int, dict] = {}
        last: dict = {}
        for ph in self.cfg.phases:
            if self._cancelled:
                break
            self._current = Runner(replace(self.cfg, phase=ph))
            for ev in self._current.events():
                if ev["type"] == "finished":
                    per_phase[ph] = last = ev["counts"]
                    yield {"type": "phase_done", "fase": ph, "counts": ev["counts"], "cancelled": ev["cancelled"]}
                else:
                    yield {**ev, "fase": ph}
        yield {"type": "finished", "counts": last, "phases": per_phase, "cancelled": self._cancelled}


def make_runner(cfg: RunConfig):
    return PhasedRunner(cfg) if cfg.mode == "fases" else Runner(cfg)
```

- [ ] **Step 4: Verificar que pasan**

Run: `.\.venv\Scripts\python -m pytest -q`
Expected: todo en verde.

- [ ] **Step 5: Commit**

```powershell
git add acfc/core/runner.py tests/test_phased_runner.py
git commit -m "feat: PhasedRunner encadena las tres fases"
```

---

### Task 8: CLI – `--modo` y `--fase`

**Files:**
- Modify: `acfc/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `make_runner`, `RunConfig` (Tasks 6–7).
- Produces: `python -m acfc <ruta> [--modo fases|unico] [--fase 1|2|3|todas] [--workers N] [--min-area X] [--force]`; por defecto `--modo fases --fase todas`. Una barra general y una por worker, recreadas en cada `start` (cada fase). Si falta la carpeta de la fase anterior imprime el mensaje en `stderr` y devuelve `2`. El resumen final muestra el conteo de cada fase.

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_cli.py`:
```python
def test_cli_por_fases_es_el_modo_por_defecto(tmp_path, make_image):
    for i in range(4):
        make_image(tmp_path / f"i{i}.jpg", size=(100, 100))
    make_image(tmp_path / "blanca.jpg", size=(100, 100), color=(255, 0, 0))
    assert main([str(tmp_path), "--workers", "2", "--backend", "fake"]) == 0
    assert len(list((tmp_path / "fase1_mapa").glob("*.jpg"))) == 5
    assert len(list((tmp_path / "fase2_personas").glob("*.jpg"))) == 4
    assert len(list((tmp_path / "outputs").glob("*.jpg"))) == 4


def test_cli_una_sola_fase(tmp_path, make_image):
    make_image(tmp_path / "a.jpg", size=(100, 100))
    assert main([str(tmp_path), "--fase", "1", "--backend", "fake"]) == 0
    assert (tmp_path / "fase1_mapa" / "a.jpg").exists() and not (tmp_path / "outputs").exists()


def test_cli_modo_unico_conserva_el_comportamiento_anterior(tmp_path, make_image):
    for i in range(3):
        make_image(tmp_path / f"i{i}.jpg")
    assert main([str(tmp_path), "--modo", "unico", "--backend", "fake"]) == 0
    assert len(list((tmp_path / "outputs").glob("*.jpg"))) == 3 and not (tmp_path / "fase1_mapa").exists()


def test_cli_fase_sin_la_carpeta_anterior_da_error_claro(tmp_path, capsys):
    assert main([str(tmp_path), "--fase", "2", "--backend", "fake"]) == 2
    assert "falta la carpeta de la fase anterior" in capsys.readouterr().err
```

- [ ] **Step 2: Verificar que fallan**

Run: `.\.venv\Scripts\python -m pytest tests/test_cli.py -q`
Expected: FAIL (`unrecognized arguments: --fase`).

- [ ] **Step 3: Implementar**

En `acfc/cli.py`:
- `build_parser`: agregar `--modo` (`choices=["fases","unico"]`, default `fases`) y `--fase` (`choices=["1","2","3","todas"]`, default `todas`).
- `main`: construir `phases = (1, 2, 3) if args.fase == "todas" else (int(args.fase),)` y `cfg = RunConfig(args.ruta, workers=..., min_area_ratio=..., force=..., backend=..., mode=args.modo, phases=phases)`; `runner = make_runner(cfg)`.
- Envolver el bucle de eventos en `try/except FileNotFoundError as exc: print(f"Error: {exc}", file=sys.stderr); return 2`.
- En el bucle, en `start`: eliminar con `prog.remove_task` las tareas anteriores (general y workers) y crear `General` (con `fase N` si `ev.get("fase")`) y `Worker i`; el resto de eventos igual.
- Resumen final: si `last.get("phases")` imprimir una línea por fase (`Fase 1: ...`), si no la línea actual.

- [ ] **Step 4: Verificar que pasan**

Run: `.\.venv\Scripts\python -m pytest -q`
Expected: todo en verde (el test CLI existente `test_cli_procesa_carpeta_con_backend_fake` pasa también en modo por fases).

- [ ] **Step 5: Commit**

```powershell
git add acfc/cli.py tests/test_cli.py
git commit -m "feat: CLI con --modo y --fase"
```

---

### Task 9: GUI – selector de modo/fase

**Files:**
- Modify: `acfc/gui.py`
- Test: `tests/test_gui_smoke.py`

**Interfaces:**
- Consumes: `make_runner`, `RunConfig`.
- Produces: `MainWindow.mode_combo` (opciones, en este orden: `Por fases (1→2→3)`, `Solo fase 1 (mapa)`, `Solo fase 2 (personas)`, `Solo fase 3 (texto)`, `Una sola pasada`); `build_config()` devuelve `RunConfig` con `mode`/`phases` según el combo; `RunnerThread` usa `make_runner`; la ventana muestra la fase actual en una etiqueta y `handle_event` ignora `phase_done`.

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_gui_smoke.py`:
```python
def test_selector_de_modo_llega_a_la_configuracion(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.path_edit.setText(str(tmp_path))
    assert win.mode_combo.count() == 5
    win.mode_combo.setCurrentIndex(0)
    cfg = win.build_config()
    assert (cfg.mode, cfg.phases) == ("fases", (1, 2, 3))
    win.mode_combo.setCurrentIndex(2)
    cfg = win.build_config()
    assert (cfg.mode, cfg.phases) == ("fases", (2,))
    win.mode_combo.setCurrentIndex(4)
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
```

- [ ] **Step 2: Verificar que fallan**

Run: `.\.venv\Scripts\python -m pytest tests/test_gui_smoke.py -q`
Expected: FAIL (`AttributeError: 'MainWindow' object has no attribute 'mode_combo'`).

- [ ] **Step 3: Implementar**

En `acfc/gui.py`:
- Importar `QComboBox` y `from acfc.core.runner import RunConfig, make_runner`.
- `RunnerThread.__init__`: `self.runner = make_runner(cfg)`.
- Crear `self.mode_combo = QComboBox()` con los 5 textos indicados; `form.addRow("Modo:", self.mode_combo)`; crear `self.phase_label = QLabel("")` y añadirlo bajo la barra general.
- `build_config`:
```python
        idx = self.mode_combo.currentIndex()
        phases = {0: (1, 2, 3), 1: (1,), 2: (2,), 3: (3,)}.get(idx, (1, 2, 3))
        return RunConfig(Path(self.path_edit.text().strip()), workers=self.workers_spin.value(),
                         force=self.force_check.isChecked(), mode="unico" if idx == 4 else "fases", phases=phases)
```
- En `handle_event`: en `start`, además de lo existente, `self.phase_label.setText(f"Fase {ev['fase']}" if ev.get("fase") else "")`; añadir rama `elif t == "phase_done": pass`; en `finished`, si `ev.get("phases")`, componer el estado con el conteo por fase.

- [ ] **Step 4: Verificar que pasan**

Run: `.\.venv\Scripts\python -m pytest -q`
Expected: todo en verde.

- [ ] **Step 5: Commit**

```powershell
git add acfc/gui.py tests/test_gui_smoke.py
git commit -m "feat: GUI con selector de modo y fase"
```

---

### Task 10: Verificación con fotos reales y corrida por fases

**Files:**
- Modify (solo si la revisión lo exige): `acfc/core/overlay.py` (`OverlayParams`), `acfc/core/phases.py`

**Interfaces:**
- Consumes: todo lo anterior.
- Produces: `fase1_mapa/`, `fase2_personas/` y `outputs/` completos para las 1932 imágenes, con sus logs y manifiestos.

- [ ] **Step 1: Suite completa antes de tocar datos reales**

Run: `.\.venv\Scripts\python -m pytest -q`
Expected: todo en verde.

- [ ] **Step 2: Fase 1 sobre las 1932 (sin GPU)**

```powershell
.\.venv\Scripts\python -m acfc "C:\Users\sergi\Documents\Nando\ACFC\ACFC" --fase 1 --workers 3 --force
```
Expected: resumen con `cropped` ≈ 1300, `copied` (sin mapa) el resto, `error` 0. Revisar `fase1_mapa\log_fase1_revisar.csv` y `log_fase1_errores.csv`.

- [ ] **Step 3: Revisar visualmente la fase 1**

Generar hojas de contacto de 15 fotos al azar de cada grupo: (a) recortadas, (b) copiadas (sin mapa). En (a) comprobar que el corte quita el mapa y no recorta contenido útil; en (b) que realmente no tienen minimapa. Si hay minimapas sin detectar o cortes donde no hay mapa, diagnosticar como en la Task 1 (componentes rojos, área, relación alto/ancho, color) y ajustar `OverlayParams`; repetir el Step 2 con `--force` hasta que la muestra salga bien. Mostrar al usuario el resumen y las hojas.

- [ ] **Step 4: Fases 2 y 3**

Antes, cerrar aplicaciones pesadas (la corrida anterior se detuvo por falta de memoria). Luego:
```powershell
.\.venv\Scripts\python -m acfc "C:\Users\sergi\Documents\Nando\ACFC\ACFC" --fase 2 --workers 2 --force
.\.venv\Scripts\python -m acfc "C:\Users\sergi\Documents\Nando\ACFC\ACFC" --fase 3 --workers 2 --force
```
`--force` en la fase 3 reemplaza las 153 salidas viejas de `outputs/` hechas con la lógica anterior. Expected: la fase 2 descarta ≈ 12–15 %; la fase 3 termina con `saved` casi total. Revisar `log_fase3_revisar.csv` y una hoja de contacto de recortes finales.

- [ ] **Step 5: Humo de la GUI**

```powershell
.\.venv\Scripts\python -m acfc
```
Expected: abre la ventana con el selector de modo; cerrar manualmente. Reportar al usuario los conteos finales por fase y los archivos a revisar.

- [ ] **Step 6: Commit**

```powershell
git add acfc
git commit -m "chore: calibracion de la fase 1 con fotos reales"
```
