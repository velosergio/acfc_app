# ACFC – Procesador de imágenes (CLI + GUI)

Fecha: 2026-10-07

## Objetivo
Dada una carpeta con imágenes (~1932, JPG/JPEG de campo con la superposición de una app de cámara GPS), generar una carpeta `outputs/` con las imágenes limpias (sin brújula, sin minimapa y sin texto), omitiendo las imágenes donde aparece una persona con uniforme blanco **y** gorra blanca.

## Alcance
- Una sola app con dos interfaces (CLI y GUI) sobre el mismo núcleo.
- Procesamiento en lotes de 5 imágenes repartidos entre N workers, con GPU (RTX 3060 6 GB).
- Barra de progreso por worker y barra de progreso general.

Fuera de alcance: inpainting, edición de las imágenes aparte del recorte, entrenamiento de modelos.

## Entradas / salidas
- Entrada: ruta a una carpeta (no recursiva) con `.jpg`/`.jpeg`/`.png`.
- `<ruta>/outputs/<mismo_nombre>.<misma_ext>`: recorte limpio.
- `<ruta>/outputs/log_descartadas.csv`: `archivo, motivo, score_gorra, score_torso`.
- `<ruta>/outputs/log_errores.csv`: imágenes ilegibles o con fallo; el proceso continúa.
- `<ruta>/outputs/log_revisar.csv`: imágenes guardadas cuyo recorte es menor al área mínima.
- Reanudable: si la salida existe se omite, salvo `--force`.

## Estructura
```
acfc_app/
  acfc/
    core/
      people.py    # YOLO personas + regla de blanco
      overlay.py   # cajas de texto (OCR), minimapa y brújula
      crop.py      # rectángulo máximo libre de cajas
      pipeline.py  # procesa 1 imagen -> descartar | guardar
      runner.py    # lotes de 5, pool de workers, cola de eventos
      events.py    # tipos de evento
    cli.py         # python -m acfc <ruta> --workers N
    gui.py         # PySide6
    __main__.py
  tests/
  requirements.txt
```
`core` no depende de ninguna interfaz. `runner` publica eventos (`worker_batch_start`, `image_done`, `discarded`, `needs_review`, `error`, `finished`) en una cola; CLI y GUI solo los consumen.

## Regla de descarte
1. YOLO detecta personas.
2. Por persona: zona cabeza (tercio superior de la caja) y zona torso; se mide la fracción de píxeles blancos en HSV (saturación baja, valor alto).
3. Se descarta la imagen si existe al menos una persona con cabeza blanca **y** torso blanco por encima de los umbrales. Una sola condición no descarta.
4. Umbrales y tamaño mínimo de persona son parámetros calibrados con las fotos reales; el log guarda los puntajes.

## Detección y recorte
- Texto: EasyOCR en GPU -> cajas de las líneas.
- Minimapa: rectángulo pegado a una esquina con color/bordes de tile de Google.
- Brújula: círculo pegado a una esquina.
- Se unen las cajas con margen y se calcula el rectángulo más grande que no intersecta ninguna caja; ese es el recorte.
- Si el área del recorte < 40 % de la imagen (configurable) se guarda igual y se anota en `log_revisar.csv`.
- Si no se detecta ninguna superposición, se copia la imagen sin cambios.

## Ejecución y progreso
- Lotes de 5; los workers toman lotes de una cola; cada worker carga YOLO y OCR una sola vez.
- Por defecto 2–3 workers (6 GB VRAM); `--workers` lo cambia.
- CLI: `rich` con una barra por worker y una general.
- GUI (PySide6): selector de carpeta, nº de workers, iniciar/cancelar, barra por worker, barra general, tabla de descartadas.
- Cancelar termina los lotes en curso y no deja archivos a medias (escritura a temporal + rename).

## Errores
Ninguna imagen detiene el proceso: los fallos se registran en `log_errores.csv`. Si CUDA no está disponible, se avisa y se usa CPU.

## Entorno
Entorno virtual con Python 3.12 en `acfc_app/.venv`: torch (CUDA), ultralytics, easyocr, opencv-python, PySide6, rich, pytest.

## Pruebas
- Unitarias (`pytest`): rectángulo máximo libre, regla de blanco, lotes de 5, formato de logs.
- Verificación visual sobre ~30 fotos reales (con persona de blanco, sin ella, overlays en distintas esquinas) antes de la corrida de 1932.
- Prueba de humo de CLI y GUI.
