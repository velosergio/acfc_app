# ACFC – Procesador de imágenes

Limpia fotos de campo que llevan la superposición de una app de cámara GPS. A partir de una carpeta de imágenes genera otra con el minimapa y el bloque de texto recortados, y deja fuera las fotos en las que aparece una persona con uniforme blanco y gorra blanca.

Acepta `.jpg`, `.jpeg` y `.png` en la carpeta indicada (no entra en subcarpetas).

## Requisitos

- Python 3.12
- Para la fase de personas y la de texto: GPU con CUDA (pensado para una RTX 3060 de 6 GB). Si CUDA no está disponible, esas fases corren en CPU y tardan mucho más.
- El modelo `yolo11s.pt` (Ultralytics lo descarga la primera vez que corre la fase de personas) y los modelos de EasyOCR (se descargan la primera vez que corre el OCR).

## Instalación

Desde la carpeta del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`ultralytics` instala PyTorch. Si esa instalación queda en CPU y tienes GPU NVIDIA, instala después la build de PyTorch con CUDA que corresponda a tu equipo: [pytorch.org](https://pytorch.org/get-started/locally/).

## Interfaz gráfica

```powershell
python -m acfc
```

También vale `python run_acfc.py` o `python -m acfc --gui`.

1. Pulsa **Examinar…** y elige la carpeta con las fotos.
2. Elige el modo (por defecto recorre las cuatro fases).
3. Ajusta **Workers** (2 es un valor razonable en 6 GB de VRAM) y el **tamaño mínimo**.
4. Pulsa **Iniciar**.

La ventana muestra una barra general, una por worker, el tiempo y la ETA de la fase, y una tabla con las imágenes descartadas. **Cancelar** detiene la corrida al terminar el lote en curso.

## Línea de comandos

```powershell
python -m acfc RUTA
```

`RUTA` es la carpeta con las fotos. Por defecto corre el modo por fases, las cuatro fases, con 2 workers.

```powershell
python -m acfc D:\fotos --workers 2
python -m acfc D:\fotos --fase 1
python -m acfc D:\fotos --sin-personas
python -m acfc D:\fotos --modo unico
python -m acfc D:\fotos --min-lado 500 --force
```

| Opción | Qué hace |
| --- | --- |
| `--modo fases` | Una carpeta por fase. Es el valor por defecto. |
| `--modo unico` | Una sola pasada: personas, mapa y texto a la vez, y el resultado va directo a `outputs/`. |
| `--fase 1\|2\|3\|4\|todas` | En modo fases, qué fase ejecutar. Por defecto `todas`. |
| `--sin-personas` | Salta la fase 2. La fase 3 lee de `fase1_mapa/`. |
| `--workers N` | Procesos en paralelo. Por defecto 2. |
| `--min-lado PX` | Descarta imágenes cuyo ancho o alto queda por debajo de estos píxeles. Por defecto 500. `0` desactiva el filtro. |
| `--min-area 0.4` | Si el recorte conserva menos de esta fracción del área, se guarda igual y se anota para revisar. |
| `--force` | Vuelve a procesar imágenes que ya están en la carpeta de salida y regenera los logs de esa fase. |

Ctrl+C cancela la corrida.

## Qué hace cada fase

El modo por fases deja el trabajo pesado (YOLO y OCR) para las fotos que ya sobrevivieron a las fases anteriores. Cada fase se puede lanzar sola, siempre que exista la carpeta de la fase previa.

| Fase | Lee | Escribe | Qué hace |
| --- | --- | --- | --- |
| 1 Mapa | la carpeta elegida | `<ruta>/fase1_mapa/` | Recorta la franja del minimapa. Si no hay mapa, copia el archivo. Con `--min-lado`, descarta aquí las que ya quedan pequeñas. |
| 2 Personas | `fase1_mapa/` | `<ruta>/fase2_personas/` | Descarta la foto si hay una persona con gorra blanca **y** uniforme blanco. El resto se copia. |
| 3 Texto | `fase2_personas/` (o `fase1_mapa/` con `--sin-personas`) | `<ruta>/outputs/` | Recorta el bloque de texto del mismo lado que el mapa. |
| 4 Tamaño | `outputs/` | borra dentro de `outputs/` | Elimina las que siguen teniendo un lado menor que `--min-lado`. |

En la interfaz, **Fases 1, 3 y 4 (sin personas)** equivale a `--sin-personas`, y **Una sola pasada** equivale a `--modo unico`.

Si interrumpes una corrida, al volver a lanzarla se omiten las imágenes que ya existen en la carpeta de salida. Marca **Rehacer las que ya existen** o pasa `--force` para repetirlas.

## Resultado y registros

Dentro de la carpeta de cada fase:

- Las imágenes procesadas, con el mismo nombre.
- `manifest.csv` en las fases 1 y 2: archivo, lado del mapa (`arriba`, `abajo` o `ninguno`) y fila de corte.
- `log_faseN_descartadas.csv`: archivo, motivo y, en la fase de personas, puntuación de gorra y torso.
- `log_faseN_revisar.csv`: recortes que quedaron por debajo del área mínima (se guardan igual).
- `log_faseN_errores.csv`: archivos que no se pudieron leer. Un fallo no detiene el resto.

En `--modo unico` esos logs se llaman `log_descartadas.csv`, `log_revisar.csv` y `log_errores.csv`, y viven en `outputs/`.

## Pruebas

```powershell
pytest
```

## Ejecutable

`ACFC.spec` empaqueta la app con PyInstaller en una carpeta (`dist\ACFC\ACFC.exe`), no en un solo archivo. Antes de construir hacen falta `yolo11s.pt` en la raíz del proyecto y los modelos de EasyOCR en `%USERPROFILE%\.EasyOCR\model` (`craft_mlt_25k.pth` y `latin_g2.pth`).

```powershell
pyinstaller ACFC.spec
```

Sin argumentos, el `.exe` abre la ventana. Con una ruta, se comporta como la línea de comandos:

```powershell
.\dist\ACFC\ACFC.exe D:\fotos --fase 1
```

Si el ejecutable se cierra por un error no controlado, el detalle queda en `acfc-error.log`, junto al `.exe`.
