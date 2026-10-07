# ACFC – Modo por fases

Fecha: 2026-10-08
Extiende: `2026-10-07-acfc-procesador-imagenes-design.md` (no lo reemplaza; el modo de una sola pasada sigue disponible como `--modo unico`).

## Motivo
La corrida de una sola pasada es lenta y pesada en memoria (cada worker carga YOLO y EasyOCR). Se reorganiza el trabajo en tres fases baratas-primero, de modo que el OCR (lo más caro) solo corra sobre las imágenes que sobreviven a las fases anteriores, y cada fase pueda inspeccionarse y relanzarse por separado.

Datos de la colección (1932 imágenes, medidos): 1264 con minimapa abajo, 36 arriba, 632 sin minimapa detectado (de una muestra de 15 sin detección, 2 sí tenían mapa y el detector falló). Altura del mapa: 12 %–38 % de la foto (mediana 23 %). El bloque de texto queda en el mismo lado vertical que el minimapa.

## Fases
Carpetas bajo `<ruta>/`: `fase1_mapa/`, `fase2_personas/`, `outputs/` (resultado final). Cada fase lee la carpeta de la anterior (la fase 1 lee `<ruta>`).

1. **Fase 1 – mapa (sin GPU).** Detecta el minimapa (por su pin). Si lo hay: corta la franja de todo el ancho a la altura del mapa (con margen), desde el borde donde está, y guarda la imagen recortada (JPEG calidad 95, escritura atómica). Si no lo hay: copia el archivo sin recomprimir. Escribe `fase1_mapa/manifest.csv` con `archivo, lado (arriba|abajo|ninguno), corte (fila)`.
2. **Fase 2 – personas (solo YOLO).** Aplica la regla vigente (persona con gorra blanca Y uniforme blanco → descartar). Las descartadas no se copian y quedan en `log_fase2_descartadas.csv` (`archivo, motivo, score_gorra, score_torso`). Las demás se copian sin recomprimir a `fase2_personas/`, junto con su fila del manifiesto (`fase2_personas/manifest.csv`, mismo formato).
3. **Fase 3 – texto (solo OCR).** Lee el manifiesto. Si hubo corte de mapa, hace OCR solo en la franja pegada a ese borde; si el lado es `ninguno`, en las dos franjas. Quita el bloque de texto con el rectángulo libre máximo ya existente (`largest_free_rect`). La brújula no se detecta ni se recorta aparte: se conserva salvo que caiga en una zona recortada. Escribe el resultado en `outputs/` con el mismo nombre. Si el recorte final queda por debajo del área mínima (40 %), se guarda y se anota en `log_fase3_revisar.csv`.

## Interfaces
- CLI: `python -m acfc <ruta> [--fase 1|2|3|todas] [--modo fases|unico] [--workers N] [--force]`. Por defecto `--modo fases --fase todas`. `--modo unico` ejecuta el pipeline actual sin cambios.
- GUI: selector de fase (Todas/1/2/3) y modo; mantiene la casilla "rehacer", una barra por worker y una general (por fase), y la tabla de descartadas.
- Eventos del runner: los mismos tipos actuales más `phase_start` (`fase`, `total`, `workers`, `batches`) y el campo `fase` en `image_done`/`finished`.
- Lotes de 5 imágenes por worker y reanudación (se omite lo que ya existe en la carpeta de salida de cada fase, salvo `--force`; `--force` reinicia los logs y manifiestos de las fases que se rehacen).

## Memoria y paralelismo
Cada fase carga solo lo que necesita: fase 1 ningún modelo (puede usar más workers), fase 2 solo YOLO, fase 3 solo EasyOCR. El número de workers aplica por fase.

## Detector de minimapa
La fase 1 depende de no fallar en fotos con mapa (la fase 3 no lo quitaría) ni cortar donde no hay mapa (se pierde el 12–38 % de la foto). Antes de dar la fase 1 por buena: corregir los fallos conocidos (`8028a0bab8273c91`, `cc6ae1bd9882ca07`), revisar visualmente muestras de las 632 sin detección y de las con corte, y comprobar la distribución de alturas de corte.

## Errores
Igual que el modo actual: ninguna imagen detiene la corrida; fallos a `log_errores.csv` (con la fase). Si falta el manifiesto al lanzar la fase 3 sola, se escanean las dos franjas.

## Pruebas
Unitarias del corte de la fase 1 (mosaico sintético arriba/abajo/ausente, margen, copia sin recomprimir), de la fase 2 (detector falso, copia sin recomprimir, log) y de la fase 3 (OCR falso, franja según manifiesto, fallback a dos franjas, área mínima). Runner por fases con backend falso, CLI, GUI (humo). Verificación visual sobre muestras reales antes de la corrida completa.

## Fuera de alcance
Detección explícita de la brújula en el modo por fases, inpainting, n8n.
