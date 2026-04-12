# Discrepancias entre el código y `paper/main.tex`

Listado exhaustivo de diferencias detectadas entre lo descrito en el paper y lo
implementado en `forest_its/`. Ordenadas por severidad (mayor → menor).

---

## 1. Severas (afectan validez de la descripción metodológica)

### 1.1 Detección de semillas: el Watershed 3D NO usa máximos 3D, usa un CHM 2D

- **Paper, §4.5.2 Paso 3 (líneas 310):** «Se identifican **máximos locales 3D de la
  densidad suavizada** como semillas (markers) del watershed.»
- **Paper, §4.5.1 (líneas 298):** «Un detector basado en raster Z-max colapsa la
  dimensión vertical y resulta sensible a estructuras intra-dosel... En este
  trabajo se utiliza en cambio un Watershed 3D volumétrico que opera sobre la
  distribución tridimensional de los puntos.»
- **Código `forest_its/segmentation/watershed3d.py:127-153`:** la detección de
  semillas se hace sobre un **CHM 2D** construido como la altura del voxel
  ocupado más alto por columna XY (`flipped = occupied[:, :, ::-1]; chm =
  argmax(flipped, axis=2)`), y luego `peak_local_max(chm_m, ...)` opera en 2D.
  Es exactamente el "raster Z-max" que el paper dice no usar. Además, el CHM se
  construye a partir de `density_grid > 0` (sin suavizar), no de
  `density_smooth` como afirma el paper.
- **Impacto:** Contradice la justificación central del método (diferencia
  Watershed 3D vs. 2D-CHM). Solo el `skimage.segmentation.watershed` final es
  volumétrico; la etapa crítica de seeding es 2D.

### 1.2 Inferencia PointNet++: el paper describe 1 pasada + 1-NN; el código hace ~1000 pasadas + KNN ponderado

- **Paper, §4.4 Inferencia (líneas 292-293):** «El plot completo se subsamplea a
  8,192 puntos preservando la distribución espacial; la red genera
  log-probabilidades por punto y la asignación a la nube original se realiza
  por **vecino más cercano**.»
- **Código `forest_its/methods/pointnet2/predict_pn2.py:168-231`:**
  - Hace `n_passes = min(1000, max(50, int(N / num_points * 3)))` subsamplings
    aleatorios del plot.
  - Acumula probabilidades por punto y las promedia.
  - Los puntos no cubiertos tras todas las pasadas se asignan por **KNN
    ponderado por inverso de distancia con k=3**, no por 1-NN.
- **Impacto:** La descripción subestima severamente la complejidad y coste de
  inferencia. Además, "preservando la distribución espacial" sugiere algo
  determinístico (e.g. FPS) pero el código usa submuestreo **aleatorio puro**.

### 1.3 Submuestreo balanceado del RF: el paper dice "por plot", el código lo hace "global"

- **Paper, §4.3 Hiperparámetros (líneas 232):** «...un submuestreo balanceado
  de hasta 500,000 puntos **por clase por plot** evita la dominancia del
  fondo.»
- **Paper, Tabla 2 (líneas 247):** «max_points_per_class | 500,000 | Submuestreo
  **por plot**, equilibrio clases»
- **Código `forest_its/methods/rf/train_rf.py:119-142`:** concatena las
  features de **todos** los plots de train en `X_train`/`y_train` y luego
  aplica el límite de 500,000 puntos por clase **una sola vez sobre el conjunto
  global**, no por plot.
- **Impacto:** El tamaño efectivo del entrenamiento es 1,000,000 puntos totales
  (500k × 2 clases) independientemente del número de plots. Con N plots de
  train esto es ~N× menos de lo que dice el paper. Afecta directamente la
  reproducibilidad.

---

## 2. Moderadas (diferencias en descripciones metodológicas concretas)

### 2.1 Augmentación PointNet++ incompleta: falta flip aleatorio en X

- **Paper, §4.4 Entrenamiento (líneas 287):** «La augmentación incluye jitter
  gaussiano (σ = 0.02 m) y escalado uniforme aleatorio en [0.9, 1.1].»
- **Código `forest_its/data/pointnet2_dataset.py:138-147`:** además de jitter y
  escalado, aplica **`xyz[:, 0] = -xyz[:, 0]` con probabilidad 0.5** (flip
  aleatorio en X). El paper no menciona este flip.

### 2.2 Las unidades del jitter son erróneas: 0.02 no es "metros"

- **Paper, §4.4 (líneas 287):** «jitter gaussiano (σ = 0.02 m)».
- **Código `forest_its/data/pointnet2_dataset.py:130-141`:** el jitter se
  aplica **después** de normalizar el subconjunto de puntos por el radio
  máximo (`xyz /= radius`), es decir, en el **espacio normalizado
  adimensional**, no en metros. σ = 0.02 corresponde a 0.02 · radius metros,
  que para un plot de ~50 m de radio son ~1 m, no 2 cm.

### 2.3 No se describe el split interno train/val dentro de `dev`

- **Paper, §3.3 (líneas 169):** «El conjunto dev se emplea para entrenamiento
  de modelos (Flujos B y C) y calibración de hiperparámetros del detector (los
  tres flujos).» Y a lo largo del paper se habla de «sobre el split **val**»
  para grid search.
- **Código `forest_its/data/splits.py:21-45` + `configs/config.yaml:7`:** el
  conjunto dev se divide internamente con `train_test_split(val_fraction=0.2,
  random_state=42)` a nivel de plot. Ese subconjunto de 20% de dev es lo que el
  código llama `val` y es donde se hace el grid search (y también la selección
  del mejor modelo durante training PN++/RF).
- **Impacto:** El paper nunca define `val`. Faltaría aclarar (i) que hay una
  partición extra 80/20 dentro de dev, (ii) que se hace a nivel de plot, y
  (iii) la semilla de aleatoriedad.

### 2.4 Fórmula de la feature "Densidad local" (feature 11)

- **Paper, Tabla 3 (línea 223):** «Densidad local | $k / V_{\text{esfera}}$»
- **Paper, §4.3 Features (línea 204):** «densidad local en una esfera de radio
  fijo».
- **Código `forest_its/preprocessing/features_rf.py:147-155`:**
  ```python
  density = float(np.sum(dists <= density_radius))
  ```
  El código devuelve el **número de vecinos (dentro del conjunto de K-NN)
  cuya distancia ≤ density_radius**, sin dividir por volumen y, sobre todo,
  **limitado por el número de vecinos K-NN (≤ 50)** — no por todos los puntos
  del plot dentro del radio.
- **Impacto:** (a) No es "k / V_esfera" (k es constante 20 o 50 por escala, el
  cociente sería constante). (b) En nubes densas (>50 pts en 0.5 m) la feature
  se satura en K, no refleja la densidad real.

### 2.5 Aumento en X (flip) no mencionado vs. descripción de "orientaciones consistentes"

- El docstring del propio dataset (`pointnet2_dataset.py:31`) dice «NO aplicar
  rotación en Z — el dataset tiene orientaciones consistentes», pero sí se
  aplica flip en X, lo cual también rompe la consistencia de orientación. El
  paper no menciona ni el razonamiento ni el flip.

### 2.6 "Ningún flujo divide la nube en bloques" vs. config pointnet2

- **Paper, §4.1 (líneas 193):** «Los tres flujos operan sobre la nube completa
  del plot. Ningún flujo divide la nube en bloques.»
- **Código `forest_its/configs/config.yaml:53-54`:** `pointnet2:` define
  `block_size: 10.0` y `block_overlap: 2.0`. Aunque `predict_pn2.py` **no** los
  usa (hace submuestreo global), estos parámetros siguen en el config y
  aparecerán en cualquier descripción del config.yaml. El paper debería
  aclararlo o eliminarse del config.

---

## 3. Menores (omisiones, numéricas puntuales, ambigüedades)

### 3.1 Gradient clipping y dropout no mencionados en Tabla 4 del paper

- **Paper, Tabla 4 (líneas 265-284):** lista hiperparámetros de entrenamiento
  PN++ pero **no menciona**:
  - `grad_clip = 1.0` (`config.yaml:44`, usado en `train_pn2.py:202`).
  - Dropout p=0.5 en el cabezal final del modelo
    (`model_msg.py:99`, `self.drop1 = nn.Dropout(0.5)`).

### 3.2 El paper menciona "invariancia al relieve" por HAG pero los "marcadores 3D" se colocan en el voxel superior de cada columna

- **Paper, §4.5.2 Paso 4:** describe un operador watershed 3D sobre una grilla
  3D regular. Eso es técnicamente cierto del operador, pero los marcadores se
  colocan en `markers[ix, iy, top_z] = tree_id`
  (`watershed3d.py:170-171`), siempre en el **top** del CHM, no donde exista
  un máximo local real de densidad 3D. Relacionado con 1.1.

### 3.3 "seg F1" reportado: ambigüedad entre F1 tree y F1 weighted

- **Paper, §5.1 (líneas 357-361):** define «seg F1» como «F1 de la clase árbol»
  con fórmula $2PR/(P+R)$ y $P=TP/(TP+FP)$, $R=TP/(TP+FN)$ sobre la clase
  árbol.
- **Código `evaluation/semantic_metrics.py:57-59`:** computa los tres (`f1_tree`,
  `f1_notree`, `f1_weighted`).
- **Logging en `run_rf_pipeline.py:324-325` y `run_pn2_pipeline.py:343-344`:**
  solo se imprime `sem_f1_weighted`, no `f1_tree`. Si el paper termina
  reportando el valor del log (f1 ponderado entre las dos clases) en la celda
  «sem F1» estaría reportando un número distinto al que define en su ecuación.

### 3.4 Mapeo de clases de fondo en el baseline: "outpoint y vegetación no clasificada" es impreciso

- **Paper, §4.2 Flujo A (líneas 197):** «...descartando únicamente las clases
  *outpoint* y **vegetación no clasificada** de FOR-instance...»
- **Código `forest_its/data/dataset.py:93-97`:** excluye las clases
  `{0: Unclassified, 3: Out-points}`. "Vegetación no clasificada" en el paper
  parece referirse a "Unclassified" (clase 0), que no es estrictamente
  vegetación; es simplemente *no clasificado*. La redacción es confusa porque
  la clase 1 en FOR-instance se llama "Low-vegetation" (vegetación baja) y
  sí entra al baseline como no-árbol.

### 3.5 Resolución del DTM no aparece en el paper

- **Paper:** no menciona la resolución de la grilla del DTM para el cómputo de
  HAG.
- **Código `configs/config.yaml:10`:** `dtm_resolution: 0.5 m`, y
  `smooth_window: 5` (uniform filter 5×5). Son parte del pipeline y deberían
  aparecer en la descripción metodológica o en el apéndice de reproducibilidad
  (Tabla 7, que actualmente está incompleta).

### 3.6 Otros hiperparámetros del grid search fijos no están completos

- **Paper, Tabla 5 (líneas 330-332):** reporta `min_tree_height = 2.0 m` y
  `min_points_per_tree = 50` como fijos para los tres flujos, lo cual coincide
  con `configs/config.yaml:62-63`. **OK**, solo se menciona aquí para
  certificar que no hay discrepancia en estos valores.

### 3.7 Paper afirma comparación con "yanx27/Pointnet_Pointnet2_pytorch" pero sin cita

- **Paper, §4.4 Arquitectura (línea 257):** «La implementación toma como base
  el repositorio **yanx27/Pointnet_Pointnet2_pytorch**.» — no aparece en la
  bibliografía. Falta citar.

### 3.8 `k_small / k_large` y `density_radius` no están documentados en el paper

- **Paper, §4.3 Features:** menciona $k = 20$ y $k = 50$ y "esfera de radio
  fijo", pero **no indica el valor** del `density_radius`.
- **Código `configs/config.yaml:19`:** `density_radius: 0.5 m`.
- **Código `configs/config.yaml:21`:** `max_neighbor_distance: 5.0 m` (cap de
  radio del KNN) tampoco aparece.

### 3.9 Normalización de intensidad "por plot" vs. por archivo

- **Paper, Tabla 3 (línea 226):** «Intensidad normalizada | $I / I_{\max}$ por
  plot».
- **Código `forest_its/data/dataset.py:51-54`:** normaliza dividiendo por el
  máximo de intensidad **del archivo .las cargado**, lo cual coincide con
  "por plot" en la práctica porque cada plot es un archivo. **Consistente**,
  se anota aquí solo para dejar registro de la verificación.

### 3.10 Split oficial: "70% dev / 30% test" sin respaldo en el metadata

- **Paper, §3.3 (líneas 169):** «El dataset prescribe un split oficial: 70% de
  los plots para desarrollo (dev) y 30% para evaluación (test)...»
- **Código `forest_its/data/dataset.py:116-157`:** lee `data_split_metadata.csv`
  y obedece ciegamente la columna `split`. El porcentaje 70/30 no se verifica
  ni es un hiperparámetro del código — es una aseveración no respaldada por
  código. Habría que contar los plots del CSV para confirmar que efectivamente
  es 70/30 tras excluir NIBIO2.

### 3.11 Hardware RTX 4050 y tiempos en el apéndice de reproducibilidad

- **Paper, Tabla 7 Appendix A (líneas 699-706):** deja pendientes campos como
  "GPU", "memoria", "tiempo total" de PointNet++, "URL repositorio" y
  "semilla aleatoria".
- **Código:** `random_state: 42` (RF y split) está en `config.yaml:6,29`; el
  hardware PN++ ya se menciona en Tabla 4 como RTX 4050 6 GB. Son campos
  completables, no errores.

### 3.12 Justificación del watershed menciona Yang et al. 2020, no citado

- **Código `watershed3d.py:29-31`:** «sigma = 0.5, Yang et al. (2020) IEEE
  JSTARS».
- **Paper:** no cita a Yang et al. 2020 en la bibliografía, a pesar de que el
  código lo usa como justificación del valor de `gaussian_sigma` por defecto.

---

## 4. Lista rápida de cambios sugeridos

Si se quisieran resolver todas estas discrepancias en el paper (sin tocar
código), las ediciones mínimas serían:

1. **§4.5.2 Paso 3 y §4.5.1**: reformular — la detección de semillas es 2D
   sobre un CHM derivado del grid 3D; el watershed se aplica luego en 3D. O
   bien: cambiar el código para hacer detección de máximos 3D reales
   (`peak_local_max(density_smooth, min_distance=..., footprint=...)`).
2. **§4.4 Inferencia PN++**: describir el esquema multi-pasada + KNN k=3, o
   cambiar el código para hacer una sola pasada + 1-NN.
3. **§4.3 y Tabla 2**: eliminar "por plot" del comentario de
   `max_points_per_class`, o cambiar el código para submuestrear por plot.
4. **§4.4 Entrenamiento**: añadir "flip aleatorio en X (p=0.5)" a la lista de
   augmentaciones; corregir unidades del jitter (es σ=0.02 en el espacio
   normalizado por el radio del sample, no en metros).
5. **§3.3**: documentar el split interno 80/20 train/val dentro de dev
   (`val_fraction=0.2`, `random_state=42`, por plot).
6. **Tabla 3 feature 11**: cambiar fórmula de "k / V_esfera" a algo coherente
   con lo que hace el código, o arreglar el código para calcular la densidad
   real.
7. **§4.4 Arquitectura**: citar el repositorio yanx27 o añadir la referencia
   a Wang et al. 2023 que ya se menciona en el docstring del modelo.
8. **Tabla 4**: añadir `grad_clip=1.0` y `dropout=0.5`.
9. **Appendix A**: completar `random_state=42`, `dtm_resolution=0.5 m`,
   `density_radius=0.5 m`, `max_neighbor_distance=5.0 m`.
10. **Bibliografía**: verificar/añadir Yang et al. (2020) IEEE JSTARS si se
    quiere respaldar `sigma=0.5` en el watershed; completar las citas marcadas
    con "[VERIFICAR...]".
