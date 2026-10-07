# ACFC Procesador de imágenes – Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** App Python (CLI + GUI sobre el mismo núcleo) que procesa una carpeta de imágenes en lotes de 5 con varios workers en GPU: descarta imágenes con persona de uniforme blanco **y** gorra blanca, recorta fuera brújula/minimapa/texto en las demás y guarda en `outputs/` con el mismo nombre.

**Architecture:** `acfc/core` contiene la lógica sin interfaz (detección de personas, detección de superposición, cálculo del recorte, pipeline por imagen, runner multiproceso que emite eventos por una cola). `cli.py` (rich) y `gui.py` (PySide6) solo consumen esos eventos.

**Tech Stack:** Python 3.12 (venv), torch CUDA, ultralytics (YOLO), easyocr, opencv-python, Pillow, PySide6, rich, pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-acfc-procesador-imagenes-design.md`

## Global Constraints

- Proyecto en `C:\Users\sergi\Documents\Nando\ACFC\acfc_app\`, fuera de la carpeta de imágenes (`C:\Users\sergi\Documents\Nando\ACFC\ACFC\`).
- Entorno virtual Python 3.12 en `acfc_app/.venv`.
- Lotes de **5** imágenes; 2–3 workers por defecto (GPU RTX 3060, 6 GB); `--workers` configurable.
- Descarte solo si hay persona con gorra blanca **Y** uniforme (torso) blanco. Una sola condición no descarta.
- Salida: `<ruta>/outputs/<mismo_nombre>.<misma_ext>`; logs `log_descartadas.csv`, `log_errores.csv`, `log_revisar.csv` en `outputs/`.
- Descartadas no se copian; quedan en log. Ningún fallo de imagen detiene el proceso.
- Recorte = rectángulo más grande sin brújula, minimapa ni texto (sin inpainting). Si área < 40 % se guarda y se anota en `log_revisar.csv`.
- Reanudable (omite salidas existentes salvo `--force`). Escritura atómica (temporal + `os.replace`).
- Barra de progreso por worker y barra general, en CLI y GUI.
- Comunicación con el usuario en español.

## Review Focus

- Imagen en escala de grises / PNG con alfa / modo CMYK → debe convertirse a RGB y procesarse (Task 5).
- Archivo corrupto o ilegible → va a `log_errores.csv` y el resto sigue (Task 5, 6).
- Nombre con tildes, espacios o `ñ` en Windows → se lee y se guarda con el mismo nombre (Task 5).
- Persona muy pequeña/lejana con ropa blanca → no debe descartar la imagen (Task 3).
- Cajas de superposición que cubren casi toda la imagen (sin rectángulo libre) → guardar original y anotar en `log_revisar.csv`, nunca fallar (Task 2, 5).
- Carpeta vacía o con menos lotes que workers → termina limpio, sin colgarse (Task 6).
- Cancelación a mitad de corrida → sin archivos `.tmp` huérfanos (Task 6).

---

### Task 1: Andamiaje, entorno y verificación de GPU

**Files:**
- Create: `acfc_app/requirements.txt`, `acfc_app/pytest.ini`, `acfc_app/.gitignore`, `acfc_app/acfc/__init__.py`, `acfc_app/acfc/core/__init__.py`, `acfc_app/tests/__init__.py`, `acfc_app/tests/conftest.py`

**Interfaces:**
- Produces: fixture `make_image(path, size=(100, 100), color=(10, 20, 30), mode="RGB")` en `tests/conftest.py`, usada por tests posteriores.

- [ ] **Step 1: Crear carpetas y archivos base**

Desde `C:\Users\sergi\Documents\Nando\ACFC\acfc_app`:

`requirements.txt`
```
numpy
pillow
opencv-python
ultralytics
easyocr
PySide6
rich
pytest
```

`pytest.ini`
```
[pytest]
pythonpath = .
testpaths = tests
```

`.gitignore`
```
.venv/
__pycache__/
*.pt
```

`acfc/__init__.py`, `acfc/core/__init__.py`, `tests/__init__.py`: archivos vacíos.

`tests/conftest.py`
```python
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def make_image():
    def _make(path, size=(100, 100), color=(10, 20, 30), mode="RGB"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, size, color).save(path)
        return path

    return _make
```

- [ ] **Step 2: Crear el venv e instalar dependencias (torch con CUDA primero)**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
```

- [ ] **Step 3: Verificar GPU y librerías**

```powershell
.\.venv\Scripts\python -c "import torch, cv2, easyocr, ultralytics, PySide6; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), cv2.__version__)"
```
Expected: `True NVIDIA GeForce RTX 3060 Laptop GPU <versión>`. Si `cv2` falla por conflicto con `opencv-python-headless` (lo trae easyocr): `pip uninstall -y opencv-python-headless opencv-python` y luego `pip install opencv-python`. Si `cuda.is_available()` es `False`, reinstalar torch con `--index-url https://download.pytorch.org/whl/cu126`.

- [ ] **Step 4: Inicializar git y commit**

```powershell
git init
git add .
git commit -m "chore: andamiaje del proyecto acfc_app"
```

---

### Task 2: Cálculo del recorte (`crop.py`)

**Files:**
- Create: `acfc/core/crop.py`
- Test: `tests/test_crop.py`

**Interfaces:**
- Produces:
  - `Box = tuple[int, int, int, int]` (x0, y0, x1, y1; x1/y1 exclusivos)
  - `pad_box(box: Box, width: int, height: int, margin: float) -> Box` (margen = `int(margin * max(width, height))`, recortado a los límites)
  - `largest_free_rect(width: int, height: int, boxes: list[Box]) -> Box | None` (`None` si no queda área libre)

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_crop.py`
```python
from acfc.core.crop import largest_free_rect, pad_box


def test_sin_cajas_devuelve_imagen_completa():
    assert largest_free_rect(100, 100, []) == (0, 0, 100, 100)


def test_una_caja_en_esquina():
    # caja arriba-izquierda: la mayor franja libre es la derecha (80x100)
    assert largest_free_rect(100, 100, [(0, 0, 20, 50)]) == (20, 0, 100, 100)


def test_mapa_abajo_izquierda_y_texto_arriba_derecha():
    boxes = [(0, 60, 30, 100), (60, 0, 100, 40)]
    assert largest_free_rect(100, 100, boxes) == (30, 40, 100, 100)


def test_cajas_cubren_todo_devuelve_none():
    assert largest_free_rect(100, 100, [(0, 0, 100, 100)]) is None


def test_pad_box_se_recorta_a_los_limites():
    assert pad_box((5, 5, 95, 95), 100, 100, 0.1) == (0, 0, 100, 100)
    assert pad_box((40, 40, 50, 50), 100, 100, 0.05) == (35, 35, 55, 55)
```

- [ ] **Step 2: Verificar que fallan**

Run: `.\.venv\Scripts\python -m pytest tests/test_crop.py -v`
Expected: FAIL (`ModuleNotFoundError: acfc.core.crop`).

- [ ] **Step 3: Implementar**

`acfc/core/crop.py`
```python
from __future__ import annotations

Box = tuple[int, int, int, int]


def pad_box(box: Box, width: int, height: int, margin: float) -> Box:
    m = int(margin * max(width, height))
    x0, y0, x1, y1 = box
    return (max(0, x0 - m), max(0, y0 - m), min(width, x1 + m), min(height, y1 + m))


def _overlaps(r: Box, b: Box) -> bool:
    return r[0] < b[2] and b[0] < r[2] and r[1] < b[3] and b[1] < r[3]


def largest_free_rect(width: int, height: int, boxes: list[Box]) -> Box | None:
    """Rectángulo de mayor área que no intersecta ninguna caja.

    Los bordes óptimos siempre coinciden con bordes de imagen o de cajas, así que
    basta probar todas las combinaciones de esas coordenadas.
    """
    xs = sorted({0, width, *(min(max(b[0], 0), width) for b in boxes), *(min(max(b[2], 0), width) for b in boxes)})
    ys = sorted({0, height, *(min(max(b[1], 0), height) for b in boxes), *(min(max(b[3], 0), height) for b in boxes)})
    best: Box | None = None
    best_area = 0
    for i, x0 in enumerate(xs):
        for x1 in xs[i + 1:]:
            for j, y0 in enumerate(ys):
                for y1 in ys[j + 1:]:
                    area = (x1 - x0) * (y1 - y0)
                    if area <= best_area:
                        continue
                    r = (x0, y0, x1, y1)
                    if not any(_overlaps(r, b) for b in boxes):
                        best, best_area = r, area
    return best
```

- [ ] **Step 4: Verificar que pasan**

Run: `.\.venv\Scripts\python -m pytest tests/test_crop.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```powershell
git add acfc/core/crop.py tests/test_crop.py
git commit -m "feat: calculo del rectangulo libre mas grande"
```

---

### Task 3: Regla de uniforme y gorra blancos (`people.py`)

**Files:**
- Create: `acfc/core/people.py`
- Test: `tests/test_people.py`

**Interfaces:**
- Consumes: `Box` de `acfc.core.crop`.
- Produces:
  - `PersonThresholds(head=0.35, torso=0.40, min_height_ratio=0.12)` (dataclass congelada)
  - `PeopleDecision(discard: bool, head: float = 0.0, torso: float = 0.0)`
  - `white_fraction(rgb: np.ndarray, s_max=50, v_min=175) -> float`
  - `decide(rgb: np.ndarray, person_boxes: list[Box], th=PersonThresholds()) -> PeopleDecision`
  - `PersonFilter(weights="yolo11s.pt", device=None, thresholds=PersonThresholds())` con `.check(rgb) -> PeopleDecision`

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_people.py`
```python
import numpy as np

from acfc.core.people import PersonThresholds, decide, white_fraction


def _img():
    return np.full((400, 200, 3), 60, dtype=np.uint8)


def _paint(img, box, fx0, fy0, fx1, fy1, value=255):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    img[int(y0 + fy0 * h):int(y0 + fy1 * h), int(x0 + fx0 * w):int(x0 + fx1 * w)] = value


def test_white_fraction():
    white = np.full((10, 10, 3), 255, dtype=np.uint8)
    dark = np.full((10, 10, 3), 40, dtype=np.uint8)
    assert white_fraction(white) == 1.0
    assert white_fraction(dark) == 0.0
    assert white_fraction(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


def test_gorra_y_torso_blancos_descarta():
    img, box = _img(), (50, 100, 150, 350)
    _paint(img, box, 0.2, 0.0, 0.8, 0.15)   # gorra
    _paint(img, box, 0.1, 0.2, 0.9, 0.55)   # torso
    assert decide(img, [box]).discard is True


def test_solo_torso_blanco_no_descarta():
    img, box = _img(), (50, 100, 150, 350)
    _paint(img, box, 0.1, 0.2, 0.9, 0.55)
    assert decide(img, [box]).discard is False


def test_solo_gorra_blanca_no_descarta():
    img, box = _img(), (50, 100, 150, 350)
    _paint(img, box, 0.2, 0.0, 0.8, 0.15)
    assert decide(img, [box]).discard is False


def test_persona_muy_pequena_se_ignora():
    img, box = _img(), (50, 100, 70, 130)  # alto 30 < 12 % de 400
    _paint(img, box, 0.0, 0.0, 1.0, 1.0)
    assert decide(img, [box], PersonThresholds()).discard is False


def test_una_persona_blanca_entre_varias_descarta():
    img = _img()
    a, b = (10, 100, 80, 350), (110, 100, 190, 350)
    _paint(img, b, 0.2, 0.0, 0.8, 0.15)
    _paint(img, b, 0.1, 0.2, 0.9, 0.55)
    assert decide(img, [a, b]).discard is True
```

- [ ] **Step 2: Verificar que fallan**

Run: `.\.venv\Scripts\python -m pytest tests/test_people.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar**

`acfc/core/people.py`
```python
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from acfc.core.crop import Box


@dataclass(frozen=True)
class PersonThresholds:
    head: float = 0.35
    torso: float = 0.40
    min_height_ratio: float = 0.12


@dataclass(frozen=True)
class PeopleDecision:
    discard: bool
    head: float = 0.0
    torso: float = 0.0


def white_fraction(rgb: np.ndarray, s_max: int = 50, v_min: int = 175) -> float:
    if rgb.size == 0:
        return 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[..., 1] <= s_max) & (hsv[..., 2] >= v_min)
    return float(mask.mean())


def _zone(rgb: np.ndarray, box: Box, fx0: float, fy0: float, fx1: float, fy1: float) -> np.ndarray:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    zx0, zx1 = int(x0 + fx0 * w), int(x0 + fx1 * w)
    zy0, zy1 = int(y0 + fy0 * h), int(y0 + fy1 * h)
    return rgb[max(zy0, 0):zy1, max(zx0, 0):zx1]


def decide(rgb: np.ndarray, person_boxes: list[Box], th: PersonThresholds = PersonThresholds()) -> PeopleDecision:
    height = rgb.shape[0]
    best = PeopleDecision(False)
    best_rank: tuple[bool, float] = (False, -1.0)
    for box in person_boxes:
        if (box[3] - box[1]) < th.min_height_ratio * height:
            continue
        head = white_fraction(_zone(rgb, box, 0.2, 0.0, 0.8, 0.15))
        torso = white_fraction(_zone(rgb, box, 0.1, 0.2, 0.9, 0.55))
        is_white = head >= th.head and torso >= th.torso
        rank = (is_white, min(head / th.head, torso / th.torso))
        if rank > best_rank:
            best, best_rank = PeopleDecision(is_white, head, torso), rank
    return best


class PersonFilter:
    def __init__(self, weights: str = "yolo11s.pt", device: str | None = None,
                 thresholds: PersonThresholds = PersonThresholds()):
        import torch
        from ultralytics import YOLO

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(weights)
        self.thresholds = thresholds

    def check(self, rgb: np.ndarray) -> PeopleDecision:
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        res = self.model.predict(bgr, classes=[0], conf=0.35, device=self.device,
                                 half=self.device == "cuda", verbose=False)[0]
        boxes = [tuple(int(v) for v in b) for b in res.boxes.xyxy.cpu().numpy()]
        return decide(rgb, boxes, self.thresholds)
```

- [ ] **Step 4: Verificar que pasan**

Run: `.\.venv\Scripts\python -m pytest tests/test_people.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```powershell
git add acfc/core/people.py tests/test_people.py
git commit -m "feat: regla de uniforme y gorra blancos con YOLO"
```

---

### Task 4: Detección de superposición (`overlay.py`)

**Files:**
- Create: `acfc/core/overlay.py`
- Test: `tests/test_overlay.py`

**Interfaces:**
- Consumes: `Box`, `pad_box` de `acfc.core.crop`.
- Produces:
  - `OverlayParams` (dataclass congelada con los umbrales listados abajo)
  - `text_boxes_from_ocr(results, width: int, y_offset: int, p: OverlayParams) -> list[Box]` (`results` en formato easyocr `[(pts, texto, conf), ...]`; devuelve 0 o 1 caja con la unión de las líneas alineadas a la derecha)
  - `find_minimap(rgb: np.ndarray, p: OverlayParams) -> list[Box]`
  - `find_compass(rgb: np.ndarray, p: OverlayParams) -> list[Box]`
  - `OverlayDetector(gpu=True, params=OverlayParams())` con `.detect(rgb) -> list[Box]` (cajas ya con margen)

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_overlay.py`
```python
import numpy as np

from acfc.core.overlay import OverlayParams, find_minimap, text_boxes_from_ocr

P = OverlayParams()


def _pts(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_texto_une_solo_lineas_alineadas_a_la_derecha():
    results = [
        (_pts(700, 10, 990, 40), "Local: 29/05/2026", 0.8),
        (_pts(800, 50, 995, 80), "Boyaca", 0.7),
        (_pts(100, 20, 300, 60), "hojas", 0.9),          # falso positivo a la izquierda
        (_pts(700, 90, 990, 120), "ruido", 0.05),        # confianza muy baja
    ]
    assert text_boxes_from_ocr(results, 1000, 0, P) == [(700, 10, 995, 80)]


def test_texto_aplica_offset_vertical_y_sin_resultados():
    results = [(_pts(800, 5, 990, 30), "Altitud", 0.9)]
    assert text_boxes_from_ocr(results, 1000, 500, P) == [(800, 505, 990, 530)]
    assert text_boxes_from_ocr([], 1000, 0, P) == []


def _fondo():
    return np.full((600, 400, 3), (90, 110, 70), dtype=np.uint8)


def test_minimapa_arriba_izquierda():
    img = _fondo()
    img[0:90, 0:120] = (197, 232, 197)
    (x0, y0, x1, y1), = find_minimap(img, P)
    assert (x0, y0) == (0, 0)
    assert abs(x1 - 120) <= 12 and abs(y1 - 90) <= 12


def test_minimapa_abajo_derecha():
    img = _fondo()
    img[510:600, 280:400] = (197, 232, 197)
    (x0, y0, x1, y1), = find_minimap(img, P)
    assert (x1, y1) == (400, 600)
    assert abs(x0 - 280) <= 12 and abs(y0 - 510) <= 12


def test_sin_minimapa_en_fondo_uniforme():
    assert find_minimap(_fondo(), P) == []


def test_cielo_nublado_no_es_minimapa():
    img = np.full((600, 400, 3), 225, dtype=np.uint8)
    assert find_minimap(img, P) == []
```

- [ ] **Step 2: Verificar que fallan**

Run: `.\.venv\Scripts\python -m pytest tests/test_overlay.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar**

`acfc/core/overlay.py`
```python
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from acfc.core.crop import Box, pad_box


@dataclass(frozen=True)
class OverlayParams:
    strip_ratio: float = 0.30        # franja superior/inferior donde se busca texto
    right_align_min: float = 0.80    # el borde derecho de cada línea debe pasar de este x relativo
    min_ocr_conf: float = 0.15
    margin: float = 0.012            # margen extra alrededor de cada caja
    tile_corner_w: float = 0.45      # zona de esquina donde se busca el minimapa
    tile_corner_h: float = 0.40
    tile_min_w: float = 0.10
    tile_min_h: float = 0.05
    tile_fill: float = 0.90          # fracción mínima de píxeles "color mapa" dentro del rectángulo
    tile_edge: float = 25.0          # contraste mínimo en los bordes internos del rectángulo
    compass_corner: float = 0.30
    compass_ring_dark: float = 0.60  # fracción mínima del anillo oscuro y poco saturado


def text_boxes_from_ocr(results, width: int, y_offset: int, p: OverlayParams) -> list[Box]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for pts, _text, conf in results:
        if conf < p.min_ocr_conf:
            continue
        px = [int(pt[0]) for pt in pts]
        py = [int(pt[1]) for pt in pts]
        if max(px) < p.right_align_min * width:
            continue
        xs0.append(min(px)); ys0.append(min(py) + y_offset)
        xs1.append(max(px)); ys1.append(max(py) + y_offset)
    if not xs0:
        return []
    return [(min(xs0), min(ys0), max(xs1), max(ys1))]


def _map_like_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return ((hsv[..., 1] <= 90) & (hsv[..., 2] >= 185)).astype(np.uint8)


def _edge_contrast(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.median(np.abs(a.astype(np.int32).mean(axis=1) - b.astype(np.int32).mean(axis=1)).sum(axis=-1)))


def _tile_in_crop(crop: np.ndarray, full_w: int, full_h: int, p: OverlayParams) -> tuple[int, int] | None:
    """Busca un rectángulo anclado en la esquina (0,0) del recorte que parezca un minimapa."""
    ch, cw = crop.shape[:2]
    step = max(2, full_w // 200)
    min_w, min_h = int(p.tile_min_w * full_w), int(p.tile_min_h * full_h)
    ws = np.arange(min_w, cw - 8, step)
    hs = np.arange(min_h, ch - 8, step)
    if len(ws) == 0 or len(hs) == 0:
        return None
    ii = cv2.integral(_map_like_mask(crop)).astype(np.float64)
    fill = ii[np.ix_(hs, ws)] / (hs[:, None] * ws[None, :])
    area = hs[:, None] * ws[None, :]
    cand = np.argwhere(fill >= p.tile_fill)
    if len(cand) == 0:
        return None
    order = np.argsort(-area[cand[:, 0], cand[:, 1]])
    for k in order[:400]:
        hi, wi = cand[k]
        w, h = int(ws[wi]), int(hs[hi])
        right = _edge_contrast(crop[:h, w - 4:w].transpose(1, 0, 2), crop[:h, w:w + 4].transpose(1, 0, 2))
        bottom = _edge_contrast(crop[h - 4:h, :w], crop[h:h + 4, :w])
        if right >= p.tile_edge and bottom >= p.tile_edge:
            return w, h
    return None


def find_minimap(rgb: np.ndarray, p: OverlayParams) -> list[Box]:
    H, W = rgb.shape[:2]
    cw, chh = int(p.tile_corner_w * W), int(p.tile_corner_h * H)
    boxes: list[Box] = []
    for right in (False, True):
        for bottom in (False, True):
            view = rgb[:, ::-1] if right else rgb
            view = view[::-1] if bottom else view
            found = _tile_in_crop(view[:chh, :cw], W, H, p)
            if not found:
                continue
            w, h = found
            x0, x1 = (W - w, W) if right else (0, w)
            y0, y1 = (H - h, H) if bottom else (0, h)
            boxes.append((x0, y0, x1, y1))
    return boxes


def find_compass(rgb: np.ndarray, p: OverlayParams) -> list[Box]:
    H, W = rgb.shape[:2]
    c = int(p.compass_corner * W)
    boxes: list[Box] = []
    for ox in (0, W - c):
        for oy in (0, H - c):
            crop = rgb[oy:oy + c, ox:ox + c]
            gray = cv2.medianBlur(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), 5)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=c, param1=120, param2=40,
                                       minRadius=int(0.04 * W), maxRadius=int(0.14 * W))
            if circles is None:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            for cx, cy, r in circles[0]:
                ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
                px = (cx + 0.93 * r * np.cos(ang)).astype(int)
                py = (cy + 0.93 * r * np.sin(ang)).astype(int)
                ok = (px >= 0) & (px < c) & (py >= 0) & (py < c)
                if ok.sum() < 20:
                    continue
                s, v = hsv[py[ok], px[ok], 1], hsv[py[ok], px[ok], 2]
                if float(((v < 120) & (s < 80)).mean()) >= p.compass_ring_dark:
                    box = (max(0, int(cx - r)) + ox, max(0, int(cy - r)) + oy,
                           min(c, int(cx + r)) + ox, min(c, int(cy + r)) + oy)
                    boxes.append(box)
                    break
    return boxes


class OverlayDetector:
    def __init__(self, gpu: bool = True, params: OverlayParams = OverlayParams()):
        import easyocr
        import torch

        self.p = params
        self.reader = easyocr.Reader(["es", "en"], gpu=gpu and torch.cuda.is_available(), verbose=False)

    def detect(self, rgb: np.ndarray) -> list[Box]:
        H, W = rgb.shape[:2]
        strip = int(self.p.strip_ratio * H)
        boxes: list[Box] = []
        for y0, y1 in ((0, strip), (H - strip, H)):
            results = self.reader.readtext(np.ascontiguousarray(rgb[y0:y1]))
            boxes += text_boxes_from_ocr(results, W, y0, self.p)
        boxes += find_minimap(rgb, self.p) + find_compass(rgb, self.p)
        return [pad_box(b, W, H, self.p.margin) for b in boxes]
```

- [ ] **Step 4: Verificar que pasan**

Run: `.\.venv\Scripts\python -m pytest tests/test_overlay.py -v`
Expected: 6 passed. Si falla un test de minimapa por tolerancia, ajustar `step`/`tile_fill` en `_tile_in_crop`, no el test.

- [ ] **Step 5: Commit**

```powershell
git add acfc/core/overlay.py tests/test_overlay.py
git commit -m "feat: deteccion de texto, minimapa y brujula"
```

---

### Task 5: Pipeline por imagen (`pipeline.py`, `backends.py`)

**Files:**
- Create: `acfc/core/pipeline.py`, `acfc/core/backends.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `largest_free_rect`, `Box` (Task 2); `PeopleDecision` (Task 3); detectores con `.check(rgb)` y `.detect(rgb)` (Tasks 3–4).
- Produces:
  - `ImageResult(name: str, status: str, reason: str = "", head: float = 0.0, torso: float = 0.0, area_ratio: float = 1.0)`; `status` ∈ `saved | discarded | review | skipped | error`
  - `process_image(path: Path, out_dir: Path, people, overlay, min_area_ratio: float = 0.4, force: bool = False) -> ImageResult` (nunca lanza excepciones)
  - `build_backend(name: str) -> tuple[people, overlay]` con `name` ∈ `"real" | "fake"`. El backend `fake` descarta si el píxel (0,0) es `(255, 0, 0)` y devuelve la caja `(0, 0, w//5, h//10)`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_pipeline.py`
```python
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
```

- [ ] **Step 2: Verificar que fallan**

Run: `.\.venv\Scripts\python -m pytest tests/test_pipeline.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar backends**

`acfc/core/backends.py`
```python
from __future__ import annotations

from acfc.core.crop import Box
from acfc.core.people import PeopleDecision


class FakePeople:
    def check(self, rgb) -> PeopleDecision:
        return PeopleDecision(discard=tuple(int(v) for v in rgb[0, 0]) == (255, 0, 0))


class FakeOverlay:
    def detect(self, rgb) -> list[Box]:
        h, w = rgb.shape[:2]
        return [(0, 0, w // 5, h // 10)]


def build_backend(name: str):
    if name == "fake":
        return FakePeople(), FakeOverlay()
    from acfc.core.overlay import OverlayDetector
    from acfc.core.people import PersonFilter

    return PersonFilter(), OverlayDetector()
```

- [ ] **Step 4: Implementar pipeline**

`acfc/core/pipeline.py`
```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from acfc.core.crop import largest_free_rect

_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}


@dataclass
class ImageResult:
    name: str
    status: str  # saved | discarded | review | skipped | error
    reason: str = ""
    head: float = 0.0
    torso: float = 0.0
    area_ratio: float = 1.0


def _save_atomic(img: Image.Image, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.tmp")
    fmt = _FORMATS.get(dest.suffix.lower(), "PNG")
    kwargs = {"quality": 95} if fmt == "JPEG" else {}
    img.save(tmp, format=fmt, **kwargs)
    os.replace(tmp, dest)


def process_image(path: Path, out_dir: Path, people, overlay,
                  min_area_ratio: float = 0.4, force: bool = False) -> ImageResult:
    dest = out_dir / path.name
    try:
        if dest.exists() and not force:
            return ImageResult(path.name, "skipped", "ya existe")
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw).convert("RGB")
        rgb = np.asarray(img)
        decision = people.check(rgb)
        if decision.discard:
            return ImageResult(path.name, "discarded", "uniforme y gorra blancos",
                               decision.head, decision.torso)
        w, h = img.size
        boxes = overlay.detect(rgb)
        if not boxes:
            rect = (0, 0, w, h)
        else:
            rect = largest_free_rect(w, h, boxes)
        if rect is None:
            _save_atomic(img, dest)
            return ImageResult(path.name, "review", "sin rectangulo libre; se guardo el original", area_ratio=1.0)
        ratio = (rect[2] - rect[0]) * (rect[3] - rect[1]) / (w * h)
        _save_atomic(img.crop(rect), dest)
        if ratio < min_area_ratio:
            return ImageResult(path.name, "review", f"recorte pequeno ({ratio:.0%})", area_ratio=ratio)
        return ImageResult(path.name, "saved", area_ratio=ratio)
    except Exception as exc:  # una imagen mala nunca detiene la corrida
        return ImageResult(path.name, "error", f"{type(exc).__name__}: {exc}")
```

- [ ] **Step 5: Verificar que pasan**

Run: `.\.venv\Scripts\python -m pytest tests/test_pipeline.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```powershell
git add acfc/core/pipeline.py acfc/core/backends.py tests/test_pipeline.py
git commit -m "feat: pipeline por imagen con escritura atomica y backends"
```

---

### Task 6: Runner multiproceso, eventos y logs (`runner.py`)

**Files:**
- Create: `acfc/core/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `process_image`, `ImageResult` (Task 5); `build_backend` (Task 5).
- Produces:
  - `RunConfig(input_dir: Path, workers: int = 2, batch_size: int = 5, min_area_ratio: float = 0.4, force: bool = False, backend: str = "real")` con propiedad `out_dir -> input_dir / "outputs"`
  - `list_images(folder: Path) -> list[Path]` (ordenadas, extensiones `.jpg/.jpeg/.png`)
  - `make_batches(items: list, size: int) -> list[list]`
  - `Runner(cfg)` con `.events() -> Iterator[dict]` y `.cancel() -> None`
  - Eventos (dicts con clave `type`):
    - `start`: `total`, `workers`, `batches`
    - `batch_start`: `worker`, `size`, `batch`
    - `image_done`: `worker` + campos de `ImageResult` (`name, status, reason, head, torso, area_ratio`)
    - `worker_error`: `worker`, `error`
    - `worker_done`: `worker`
    - `finished`: `counts` (dict status → n), `cancelled`
  - Logs en `out_dir`: `log_descartadas.csv` (`archivo,motivo,score_gorra,score_torso`), `log_errores.csv` (`archivo,error`), `log_revisar.csv` (`archivo,area_ratio,motivo`).

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_runner.py`
```python
import csv

from acfc.core.runner import RunConfig, Runner, list_images, make_batches


def _collect(cfg):
    runner = Runner(cfg)
    return runner, list(runner.events())


def test_make_batches_de_5():
    assert make_batches(list(range(12)), 5) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]
    assert make_batches([], 5) == []


def test_list_images_filtra_extensiones(tmp_path, make_image):
    make_image(tmp_path / "b.jpg"); make_image(tmp_path / "a.PNG"); (tmp_path / "x.txt").write_text("x")
    assert [p.name for p in list_images(tmp_path)] == ["a.PNG", "b.jpg"]


def test_corrida_completa_con_backend_fake(tmp_path, make_image):
    for i in range(7):
        make_image(tmp_path / f"ok{i}.jpg")
    make_image(tmp_path / "blanca.jpg", color=(255, 0, 0))
    (tmp_path / "rota.jpg").write_bytes(b"basura")
    cfg = RunConfig(tmp_path, workers=2, backend="fake")
    _, events = _collect(cfg)

    assert events[0]["type"] == "start" and events[0]["total"] == 9
    assert events[-1]["type"] == "finished"
    counts = events[-1]["counts"]
    assert counts == {"saved": 7, "discarded": 1, "error": 1}
    assert {e["worker"] for e in events if e["type"] == "worker_done"} == {0, 1}
    assert max(e["size"] for e in events if e["type"] == "batch_start") <= 5

    out = tmp_path / "outputs"
    assert sorted(p.name for p in out.glob("*.jpg")) == sorted(f"ok{i}.jpg" for i in range(7))
    desc = list(csv.DictReader((out / "log_descartadas.csv").open(encoding="utf-8")))
    assert [r["archivo"] for r in desc] == ["blanca.jpg"]
    err = list(csv.DictReader((out / "log_errores.csv").open(encoding="utf-8")))
    assert [r["archivo"] for r in err] == ["rota.jpg"]


def test_carpeta_vacia_termina_limpio(tmp_path):
    _, events = _collect(RunConfig(tmp_path, workers=3, backend="fake"))
    assert [e["type"] for e in events] == ["start", "finished"]
    assert events[-1]["counts"] == {}


def test_menos_lotes_que_workers(tmp_path, make_image):
    make_image(tmp_path / "a.jpg")
    _, events = _collect(RunConfig(tmp_path, workers=4, backend="fake"))
    assert events[0]["workers"] == 1
    assert events[-1]["counts"] == {"saved": 1}


def test_cancelar_antes_de_empezar_no_deja_tmp(tmp_path, make_image):
    for i in range(10):
        make_image(tmp_path / f"i{i}.jpg")
    runner = Runner(RunConfig(tmp_path, workers=2, backend="fake"))
    runner.cancel()
    events = list(runner.events())
    assert events[-1]["type"] == "finished" and events[-1]["cancelled"] is True
    assert not list((tmp_path / "outputs").glob(".*.tmp"))
```

- [ ] **Step 2: Verificar que fallan**

Run: `.\.venv\Scripts\python -m pytest tests/test_runner.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar**

`acfc/core/runner.py`
```python
from __future__ import annotations

import csv
import multiprocessing as mp
import queue
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from acfc.core.backends import build_backend
from acfc.core.pipeline import process_image

EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class RunConfig:
    input_dir: Path
    workers: int = 2
    batch_size: int = 5
    min_area_ratio: float = 0.4
    force: bool = False
    backend: str = "real"

    @property
    def out_dir(self) -> Path:
        return Path(self.input_dir) / "outputs"


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def make_batches(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def worker_main(worker_id: int, task_q, event_q, cfg: RunConfig, cancel_ev) -> None:
    try:
        people, overlay = build_backend(cfg.backend)
    except Exception as exc:
        event_q.put({"type": "worker_error", "worker": worker_id, "error": f"{type(exc).__name__}: {exc}"})
        event_q.put({"type": "worker_done", "worker": worker_id})
        return
    n = 0
    while True:
        batch = task_q.get()
        if batch is None or cancel_ev.is_set():
            break
        n += 1
        event_q.put({"type": "batch_start", "worker": worker_id, "size": len(batch), "batch": n})
        for p in batch:
            if cancel_ev.is_set():
                break
            res = process_image(Path(p), cfg.out_dir, people, overlay, cfg.min_area_ratio, cfg.force)
            event_q.put({"type": "image_done", "worker": worker_id, **asdict(res)})
    event_q.put({"type": "worker_done", "worker": worker_id})


class _Logs:
    FILES = {
        "discarded": ("log_descartadas.csv", ["archivo", "motivo", "score_gorra", "score_torso"]),
        "error": ("log_errores.csv", ["archivo", "error"]),
        "review": ("log_revisar.csv", ["archivo", "area_ratio", "motivo"]),
    }

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def write(self, ev: dict) -> None:
        spec = self.FILES.get(ev["status"])
        if not spec:
            return
        name, header = spec
        path = self.out_dir / name
        new = not path.exists()
        row = {
            "discarded": [ev["name"], ev["reason"], f'{ev["head"]:.3f}', f'{ev["torso"]:.3f}'],
            "error": [ev["name"], ev["reason"]],
            "review": [ev["name"], f'{ev["area_ratio"]:.3f}', ev["reason"]],
        }[ev["status"]]
        with path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(header)
            w.writerow(row)


class Runner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self._ctx = mp.get_context("spawn")
        self._cancel = self._ctx.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def events(self) -> Iterator[dict]:
        cfg = self.cfg
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        files = list_images(cfg.input_dir)
        batches = make_batches([str(p) for p in files], cfg.batch_size)
        n_workers = min(cfg.workers, len(batches))
        yield {"type": "start", "total": len(files), "workers": n_workers, "batches": len(batches)}

        counts: Counter = Counter()
        if n_workers:
            task_q, event_q = self._ctx.Queue(), self._ctx.Queue()
            for b in batches:
                task_q.put(b)
            for _ in range(n_workers):
                task_q.put(None)
            procs = [self._ctx.Process(target=worker_main, args=(i, task_q, event_q, cfg, self._cancel), daemon=True)
                     for i in range(n_workers)]
            for p in procs:
                p.start()
            logs = _Logs(cfg.out_dir)
            finished: set[int] = set()
            dead_polls = [0] * n_workers
            while len(finished) < n_workers:
                try:
                    ev = event_q.get(timeout=0.5)
                except queue.Empty:
                    for i, p in enumerate(procs):
                        if i in finished:
                            continue
                        dead_polls[i] = dead_polls[i] + 1 if not p.is_alive() else 0
                        if dead_polls[i] >= 3:
                            finished.add(i)
                            yield {"type": "worker_error", "worker": i, "error": "el worker termino inesperadamente"}
                            yield {"type": "worker_done", "worker": i}
                    continue
                if ev["type"] == "image_done":
                    counts[ev["status"]] += 1
                    logs.write(ev)
                elif ev["type"] == "worker_done":
                    finished.add(ev["worker"])
                yield ev
            for p in procs:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
        yield {"type": "finished", "counts": dict(counts), "cancelled": self._cancel.is_set()}
```

- [ ] **Step 4: Verificar que pasan**

Run: `.\.venv\Scripts\python -m pytest tests/test_runner.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```powershell
git add acfc/core/runner.py tests/test_runner.py
git commit -m "feat: runner multiproceso con lotes de 5, eventos y logs"
```

---

### Task 7: Interfaz CLI (`cli.py`, `__main__.py`)

**Files:**
- Create: `acfc/cli.py`, `acfc/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Runner`, `RunConfig`, eventos (Task 6).
- Produces: `main(argv: list[str] | None = None) -> int` en `acfc.cli`; `python -m acfc <ruta> [--workers N] [--min-area 0.4] [--force]` ejecuta la CLI; `python -m acfc` sin argumentos (o con `--gui`) abre la GUI (Task 8). Opción oculta `--backend {real,fake}` para pruebas.

- [ ] **Step 1: Escribir el test que falla**

`tests/test_cli.py`
```python
from acfc.cli import main


def test_cli_procesa_carpeta_con_backend_fake(tmp_path, make_image):
    for i in range(6):
        make_image(tmp_path / f"i{i}.jpg")
    make_image(tmp_path / "blanca.jpg", color=(255, 0, 0))
    code = main([str(tmp_path), "--workers", "2", "--backend", "fake"])
    assert code == 0
    assert len(list((tmp_path / "outputs").glob("*.jpg"))) == 6


def test_cli_ruta_inexistente_devuelve_error(tmp_path, capsys):
    assert main([str(tmp_path / "nada"), "--backend", "fake"]) == 2
    assert "no existe" in capsys.readouterr().err.lower()
```

- [ ] **Step 2: Verificar que falla**

Run: `.\.venv\Scripts\python -m pytest tests/test_cli.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar**

`acfc/cli.py`
```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from acfc.core.runner import RunConfig, Runner


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="acfc", description="Descarta imagenes con uniforme+gorra blancos y recorta mapa/texto.")
    ap.add_argument("ruta", type=Path, help="carpeta con las imagenes")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-area", type=float, default=0.4, help="area minima del recorte (0-1) antes de marcar para revisar")
    ap.add_argument("--force", action="store_true", help="rehacer imagenes que ya existen en outputs")
    ap.add_argument("--backend", choices=["real", "fake"], default="real", help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ruta.is_dir():
        print(f"Error: la carpeta no existe: {args.ruta}", file=sys.stderr)
        return 2
    cfg = RunConfig(args.ruta, workers=max(1, args.workers), min_area_ratio=args.min_area,
                    force=args.force, backend=args.backend)
    runner = Runner(cfg)
    columns = (TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn())
    worker_tasks: dict[int, int] = {}
    last = {}
    try:
        with Progress(*columns) as prog:
            overall = prog.add_task("General", total=0)
            for ev in runner.events():
                t = ev["type"]
                if t == "start":
                    prog.update(overall, total=ev["total"])
                    for w in range(ev["workers"]):
                        worker_tasks[w] = prog.add_task(f"Worker {w}", total=0)
                elif t == "batch_start":
                    prog.reset(worker_tasks[ev["worker"]], total=ev["size"],
                               description=f"Worker {ev['worker']} lote {ev['batch']}")
                elif t == "image_done":
                    prog.advance(worker_tasks[ev["worker"]])
                    prog.advance(overall)
                elif t == "worker_error":
                    prog.console.print(f"[red]Worker {ev['worker']}: {ev['error']}[/red]")
                elif t == "finished":
                    last = ev
    except KeyboardInterrupt:
        runner.cancel()
        print("Cancelando...", file=sys.stderr)
        return 130
    counts = last.get("counts", {})
    print(f"Listo. guardadas={counts.get('saved', 0)} revisar={counts.get('review', 0)} "
          f"descartadas={counts.get('discarded', 0)} errores={counts.get('error', 0)} "
          f"omitidas={counts.get('skipped', 0)}  ->  {cfg.out_dir}")
    return 0
```

`acfc/__main__.py`
```python
import sys


def _main() -> int:
    argv = sys.argv[1:]
    if not argv or "--gui" in argv:
        from acfc.gui import run_gui

        return run_gui()
    from acfc.cli import main

    return main(argv)


if __name__ == "__main__":
    sys.exit(_main())
```

- [ ] **Step 4: Verificar que pasa**

Run: `.\.venv\Scripts\python -m pytest tests/test_cli.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add acfc/cli.py acfc/__main__.py tests/test_cli.py
git commit -m "feat: interfaz CLI con barras por worker y general"
```

---

### Task 8: Interfaz GUI (`gui.py`)

**Files:**
- Create: `acfc/gui.py`
- Test: `tests/test_gui_smoke.py`

**Interfaces:**
- Consumes: `Runner`, `RunConfig`, eventos (Task 6).
- Produces: `MainWindow` con `handle_event(ev: dict)` (recibe los eventos del runner) y `run_gui() -> int`.

- [ ] **Step 1: Escribir el test de humo que falla**

`tests/test_gui_smoke.py`
```python
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from acfc.gui import MainWindow


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
```

- [ ] **Step 2: Verificar que falla**

Run: `.\.venv\Scripts\python -m pytest tests/test_gui_smoke.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar**

`acfc/gui.py`
```python
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QApplication, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from acfc.core.runner import RunConfig, Runner


class RunnerThread(QThread):
    event = Signal(dict)

    def __init__(self, cfg: RunConfig):
        super().__init__()
        self.runner = Runner(cfg)

    def run(self) -> None:
        for ev in self.runner.events():
            self.event.emit(ev)

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
        form = QFormLayout()
        form.addRow("Carpeta:", row)
        form.addRow("Workers:", self.workers_spin)

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
        self.workers_box = QGroupBox("Workers")
        self.workers_layout = QVBoxLayout(self.workers_box)
        self.status = QLabel("Listo.")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Archivo", "Motivo", "Gorra", "Torso"])
        self.table.horizontalHeader().setStretchLastSection(True)

        central = QWidget()
        lay = QVBoxLayout(central)
        for item in (form, buttons):
            lay.addLayout(item)
        lay.addWidget(self.overall)
        lay.addWidget(self.workers_box)
        lay.addWidget(self.status)
        lay.addWidget(QLabel("Imágenes descartadas:"))
        lay.addWidget(self.table)
        self.setCentralWidget(central)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Carpeta de imágenes")
        if d:
            self.path_edit.setText(d)

    def _start(self) -> None:
        folder = Path(self.path_edit.text().strip())
        if not folder.is_dir():
            QMessageBox.warning(self, "Carpeta", "La carpeta no existe.")
            return
        self.table.setRowCount(0)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status.setText("Procesando…")
        self.thread_ = RunnerThread(RunConfig(folder, workers=self.workers_spin.value()))
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
            if ev["status"] == "discarded":
                r = self.table.rowCount()
                self.table.insertRow(r)
                for c, v in enumerate([ev["name"], ev["reason"], f'{ev["head"]:.2f}', f'{ev["torso"]:.2f}']):
                    self.table.setItem(r, c, QTableWidgetItem(v))
        elif t == "worker_error":
            self.status.setText(f"Error en worker {ev['worker']}: {ev['error']}")
        elif t == "finished":
            c = ev["counts"]
            estado = "Cancelado" if ev["cancelled"] else "Terminado"
            self.status.setText(f"{estado}. guardadas={c.get('saved', 0)} revisar={c.get('review', 0)} "
                                f"descartadas={c.get('discarded', 0)} errores={c.get('error', 0)}")
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

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
```

- [ ] **Step 4: Verificar que pasa**

Run: `.\.venv\Scripts\python -m pytest tests/test_gui_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 5: Suite completa y commit**

Run: `.\.venv\Scripts\python -m pytest -v`
Expected: todos pasan.

```powershell
git add acfc/gui.py tests/test_gui_smoke.py
git commit -m "feat: interfaz GUI con barras por worker y general"
```

---

### Task 9: Calibración con imágenes reales y corrida final

**Files:**
- Create: `tools/debug_overlay.py`
- Modify (solo constantes si hace falta): `acfc/core/overlay.py` (`OverlayParams`), `acfc/core/people.py` (`PersonThresholds`, zonas)

**Interfaces:**
- Consumes: `build_backend("real")`, `largest_free_rect`, `decide` (Tasks 2–5).
- Produces: `python tools/debug_overlay.py <carpeta> --n 30 --out <dir>` que guarda por cada imagen de muestra una versión anotada (cajas detectadas en rojo, recorte final en verde, texto con el veredicto de personas).

- [ ] **Step 1: Escribir la herramienta de depuración**

`tools/debug_overlay.py`
```python
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image, ImageOps

from acfc.core.backends import build_backend
from acfc.core.crop import largest_free_rect
from acfc.core.runner import list_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    people, overlay = build_backend("real")
    files = list_images(a.folder)
    random.Random(a.seed).shuffle(files)
    for p in files[: a.n]:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        rgb = np.asarray(img)
        h, w = rgb.shape[:2]
        dec = people.check(rgb)
        boxes = overlay.detect(rgb)
        rect = largest_free_rect(w, h, boxes) if boxes else (0, 0, w, h)
        vis = np.ascontiguousarray(rgb[..., ::-1]).copy()
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 4)
        if rect:
            cv2.rectangle(vis, (rect[0], rect[1]), (rect[2] - 1, rect[3] - 1), (0, 255, 0), 4)
        cv2.putText(vis, f"descartar={dec.discard} gorra={dec.head:.2f} torso={dec.torso:.2f}",
                    (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, w / 900, (255, 0, 255), 3)
        cv2.imwrite(str(a.out / p.name), vis)
        print(p.name, "descartar" if dec.discard else "ok", f"{dec.head:.2f}/{dec.torso:.2f}", "cajas:", len(boxes))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Ejecutar sobre una muestra de 30 imágenes**

```powershell
.\.venv\Scripts\python tools\debug_overlay.py "C:\Users\sergi\Documents\Nando\ACFC\ACFC" --n 30 --out "$env:TEMP\acfc_debug"
```
Expected: 30 líneas de resultado; la primera ejecución descarga `yolo11s.pt` y los modelos de EasyOCR (requiere internet).

- [ ] **Step 3: Revisar visualmente las anotadas y ajustar constantes**

Abrir varias imágenes de `$env:TEMP\acfc_debug` y comprobar: (a) las cajas rojas cubren texto, minimapa y brújula sin cubrir contenido real; (b) el rectángulo verde es el recorte esperado; (c) el veredicto de personas coincide (persona con gorra y uniforme blancos = descartar). Si hay falsos positivos/negativos, ajustar solo las constantes de `OverlayParams` / `PersonThresholds` (o las fracciones de las zonas en `people._zone`) y repetir el Step 2 con otra `--seed` hasta que la muestra salga bien. Mostrar al usuario un resumen de la muestra revisada y los valores finales.

- [ ] **Step 4: Re-ejecutar la suite**

Run: `.\.venv\Scripts\python -m pytest -v`
Expected: todos pasan (si cambió algún umbral, ajustar los tests sintéticos afectados de forma coherente).

- [ ] **Step 5: Corrida final sobre las 1932 imágenes (CLI)**

```powershell
.\.venv\Scripts\python -m acfc "C:\Users\sergi\Documents\Nando\ACFC\ACFC" --workers 2
```
Expected: barras por worker y general hasta 1932; resumen final; `outputs/` con las imágenes limpias y los CSV. Revisar `log_revisar.csv` y `log_errores.csv` y reportar los conteos al usuario.

- [ ] **Step 6: Humo de la GUI y commit**

```powershell
.\.venv\Scripts\python -m acfc
```
Expected: abre la ventana; cerrar manualmente. Luego:

```powershell
git add tools/debug_overlay.py acfc
git commit -m "feat: herramienta de depuracion y calibracion de umbrales"
```
