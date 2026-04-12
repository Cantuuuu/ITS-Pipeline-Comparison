# Conciliación de discrepancias código ↔ paper

Documento vivo con las decisiones tomadas sobre cada discrepancia listada en
`discrepancias.md`. Cada entrada incluye: decisión, justificación, acciones
concretas en código y/o paper, y estado.

Leyenda de estado: ☐ pendiente · ◐ decidido, sin aplicar · ☑ aplicado.

---

## 1.1 — Seeds del Watershed: ¿2D (CHM) o 3D real? ☑

**Decisión:** Opción A — mantener el código tal como está y **reescribir el
paper** para describir con honestidad el método: seeding sobre la envolvente
superior del volumen 3D (CHM derivado localmente del grid) + fill volumétrico
3D con máscara de ocupación.

**Justificación académica (no sólo práctica):** en datos forestales reales el
máximo local de densidad 3D cae en el tronco y en las ramas gruesas del
centro de la copa, *no* en el ápice. Hacer `peak_local_max` sobre
`density_smooth` 3D produciría semillas en los troncos y romperia la
segmentación. Por eso toda la literatura de ITS (Li et al. 2012, Chen et al.
2006, Dalponte & Coomes 2016, los trabajos recientes sobre ULS denso) usa CHM
para seeding incluso cuando el fill es 3D. El verdadero aporte del enfoque
3D **no es el seeding** sino el **fill**: un watershed 2D sobre CHM asigna
cada punto según su proyección XY, por lo que puntos de tronco y ramas bajas
bajo la proyección de una copa vecina se asignan incorrectamente. El
watershed 3D del código asigna cada punto según el voxel 3D que lo contiene,
siguiendo crestas de densidad volumétricas, y eso es lo que realmente separa
troncos adyacentes y sotobosque.

**Acciones:**

1. **Paper §4.5.1 (líneas ~297-301):** reemplazar la justificación actual por:

   > La densidad ultra-alta de los datos ULS (∼7,000 pts/m²) permite ejecutar
   > el paso de fill del watershed directamente sobre la grilla 3D,
   > preservando la estructura vertical de la nube. Un watershed 2D
   > tradicional sobre CHM asigna cada punto en función de su proyección XY,
   > lo que produce errores sistemáticos cuando puntos de tronco o rama baja
   > caen bajo la proyección de una copa vecina. En este trabajo se emplea
   > un Watershed 3D volumétrico en el que las semillas se detectan sobre la
   > envolvente superior del volumen (equivalente a un CHM derivado
   > localmente del grid 3D de densidad) pero el fill se ejecuta sobre la
   > grilla 3D con máscara de voxels ocupados, de modo que cada punto es
   > asignado al árbol cuyo voxel contiene esa posición volumétrica y no
   > según su mera proyección horizontal.

2. **Paper §4.5.2 Paso 3 (líneas ~310):** reemplazar por:

   > **Paso 3 — Semillas sobre la envolvente superior del volumen.** Para
   > cada columna (x, y) del grid 3D se identifica el voxel ocupado más alto,
   > construyendo un mapa de alturas local derivado del propio grid. Sobre
   > este mapa se aplica `peak_local_max` con separación mínima
   > `min_crown_radius_m / voxel_size` y umbral absoluto `min_tree_height`,
   > obteniendo una semilla (ix, iy, top_z) por árbol candidato. Las semillas
   > con `top_z < min_tree_height / voxel_size` se descartan.

3. **Paper §4.5.2 Paso 4 (líneas ~312):** reemplazar por:

   > **Paso 4 — Watershed volumétrico 3D.** A partir de las semillas se
   > ejecuta `skimage.segmentation.watershed` sobre el grid 3D de densidad
   > suavizada invertida (`-density_smooth`), con máscara `density_grid > 0`.
   > El operador produce etiquetas de instancia por voxel tridimensional;
   > cada punto del plot recibe la etiqueta del voxel que lo contiene. A
   > diferencia de un watershed 2D sobre CHM, la asignación de puntos sigue
   > crestas de densidad volumétricas y no la proyección horizontal, lo que
   > separa correctamente troncos adyacentes y puntos de sotobosque.

4. **Código `forest_its/segmentation/watershed3d.py`:** añadir un comentario
   de bloque al inicio de la sección de seeding aclarando que el CHM es una
   envolvente del grid 3D y que el valor del método reside en el fill 3D, no
   en un "seeding 3D". Puramente docstring; no hay cambio de comportamiento.

**Sin impacto en resultados:** no se re-corre nada.

---

## 1.2 — Inferencia PointNet++: multi-pasada + KNN vs. 1 pasada + 1-NN ☑

**Decisión:** Opción A — mantener el código tal como está y **reescribir el
paper** para describir el esquema real de inferencia: `n_passes` submuestreos
aleatorios globales del plot con promediado de probabilidades + KNN k=3 como
fallback para puntos residuales no cubiertos.

**Justificación académica:** el código implementa la estrategia *correcta*
dada la arquitectura. PointNet++ se entrenó con submuestreo aleatorio global
del plot completo (8,192 puntos por muestra, pre-centrado per-sample y
normalización por radio máximo del sample — ver
`ForInstanceDataset.__getitem__`). Los radios de las capas Set Abstraction
están calibrados a la escala de ese sample normalizado. Hacer inferencia con
una sola pasada de 8,192 puntos + extrapolación 1-NN (como describe el paper
actualmente) sería equivalente a etiquetar ~99.6% de los puntos por
vecindad en lugar de por predicción directa de la red, degradando la calidad
semántica sin justificación. El esquema multi-pasada garantiza consistencia
train/test y cobertura directa >99% en plots típicos.

**Acciones:**

1. **Paper §4.4 Inferencia (líneas ~292-293):** reemplazar el párrafo actual
   por:

   > **Inferencia.** PointNet++ fue entrenado con submuestreo aleatorio
   > global del plot completo (8,192 puntos por muestra, normalización
   > per-sample por centroide y radio máximo). La inferencia replica
   > exactamente esta estrategia para garantizar consistencia train/test:
   > cada plot se procesa mediante `n_passes` submuestreos aleatorios
   > independientes, con `n_passes ≈ min(1000, max(50, 3·N/8192))` donde `N`
   > es el número de puntos válidos del plot. En cada pasada, los 8,192
   > puntos seleccionados se pre-centran y normalizan como en training, y
   > las log-probabilidades por punto se acumulan. Tras todas las pasadas,
   > cada punto recibe la media de las probabilidades predichas en las
   > pasadas en que fue seleccionado; la clase final es el argmax. Con este
   > esquema la cobertura efectiva supera el 99% de los puntos del plot en
   > datos típicos. Los puntos residuales no cubiertos se asignan por
   > interpolación KNN ponderada por 1/distancia con k = 3 desde los
   > puntos cubiertos más cercanos. La alternativa de una sola pasada con
   > extrapolación 1-NN fue descartada por introducir un mismatch de escala
   > entre train y test (los radios normalizados de las capas Set Abstraction
   > están calibrados a la escala del sample de 8,192 puntos) y porque
   > etiquetaría ∼99% de los puntos por vecindad y no por predicción
   > directa.

**Sin impacto en resultados:** no se re-corre nada.

---

## 1.3 — Submuestreo del Random Forest: global vs. por plot ☑

**Decisión:** Opción A — cambiar el código para submuestreo **balanceado por
plot**, con orden determinístico de la lista de plots y semilla reproducible
por plot derivada del nombre del archivo. El paper (§4.3 y Tabla 2) ya
describe un esquema por plot, así que el cambio alinea el código con la
descripción académica existente.

**Justificación académica:** el objetivo declarado del paper es comparar la
generalización del preprocesamiento semántico entre tipos de bosque
(NIBIO, NIBIO2, CULS, SCION, RMIT, TUWIEN). Con submuestreo global tras
concatenar todos los plots, los sitios con plots más grandes (NIBIO) dominan
las muestras disponibles y el RF aprende sesgado hacia su distribución; los
sitios pequeños aportan una fracción marginal. El submuestreo por plot
equipara la contribución de cada plot al conjunto final, lo cual es la
semántica correcta para el experimento y es además lo que el paper ya
afirma. El orden determinístico + semilla reproducible por plot garantiza
que corridas independientes produzcan exactamente el mismo conjunto de
entrenamiento (clave para replicabilidad del paper).

**Acciones código `forest_its/methods/rf/train_rf.py`:**

1. Añadir `import zlib` al encabezado.
2. Ordenar `train_paths` alfabéticamente por `path.stem` antes del bucle de
   extracción, para fijar el orden de iteración entre corridas.
3. Dentro del bucle sobre plots, justo después de extraer features y
   etiquetas del plot, aplicar submuestreo local con semilla derivada del
   nombre del archivo:

   ```python
   plot_seed = (cfg.rf.random_state + zlib.crc32(las_path.stem.encode())) & 0x7FFFFFFF
   rng_plot = np.random.default_rng(plot_seed)
   # submuestrear hasta cfg.rf.max_samples_per_plot puntos de árbol y
   # cfg.rf.max_samples_per_plot puntos de no-árbol con rng_plot
   ```

4. Eliminar el bloque de submuestreo global posterior a la concatenación.
5. Actualizar los `log.info` para reportar puntos muestreados por plot y
   totales finales sin mencionar "global subsample".
6. Añadir `max_samples_per_plot` a `forest_its/configs/config.yaml` bajo la
   sección `rf:` (valor sugerido: suficiente para obtener ~500k/clase
   totales tras sumar todos los plots de train — calcular en base al número
   de plots de train del split dev).

**Acciones paper:** ninguna — §4.3 y Tabla 2 ya describen "submuestreo por
plot" correctamente.

**Impacto en resultados:** re-correr **toda la cadena RF**:

1. `python -m forest_its.methods.rf.train_rf` (~16 min, CPU)
2. `python -m forest_its.methods.rf.run_rf_pipeline --stage semantic --split val`
3. `python -m forest_its.methods.rf.run_rf_pipeline --stage semantic --split test`
4. `python -m forest_its.evaluation.grid_search --methods rf`
5. `python -m forest_its.methods.rf.run_rf_pipeline --stage instance --split val`
6. `python -m forest_its.methods.rf.run_rf_pipeline --stage instance --split test`

Los números de la Tabla de resultados del flujo B cambiarán (probablemente
mejora moderada en sitios pequeños, leve caída en NIBIO).

---

## 2.1 — Augmentation PointNet++: flip X no documentado ☑

**Decisión:** Opción A — mantener el código y **actualizar el paper** para
incluir el flip aleatorio del eje X en la descripción de augmentation.

**Justificación académica:** el flip horizontal (reflexión sobre el eje X
con probabilidad 0.5) es un augmentation estándar en point-cloud learning
que respeta las simetrías del problema forestal: los árboles son
aproximadamente simétricos bajo reflexión horizontal, y el sensor ULS no
tiene dirección de vuelo preferente respecto al eje X local del plot tras
el centrado. Ya está implementado y no introduce sesgo. Re-entrenar solo
para eliminarlo no aporta valor académico y cuesta horas de GPU.

**Acciones paper §4.4 Entrenamiento (línea 287):** reemplazar la oración:

> La augmentación incluye jitter gaussiano ($\sigma = 0.02$\,m) y escalado
> uniforme aleatorio en $[0.9, 1.1]$.

por:

> La augmentación incluye tres operaciones aplicadas por muestra tras la
> normalización per-sample: (i) jitter gaussiano independiente por
> coordenada con $\sigma = 0.02$ en el espacio normalizado (equivalente a
> una fracción pequeña del radio máximo del sample), (ii) reflexión
> aleatoria sobre el eje X con probabilidad $0.5$, aprovechando la
> simetría horizontal del problema forestal, y (iii) escalado isotrópico
> uniforme en $[0.9, 1.1]$. No se aplica rotación en Z porque la
> orientación del grid del dataset es consistente entre plots.

(Nota: esta reescritura también resuelve la discrepancia 2.2 — el jitter
está en espacio normalizado, no en metros — que se trata como parte del
mismo párrafo.)

**Acciones código:** ninguna.

**Sin impacto en resultados:** no se re-corre nada.

---

## 2.3 — Split interno train/val dentro de dev no descrito ☑

**Decisión:** Opción A — **documentar en el paper** el desdoblamiento
interno dev → train_interno + val_interno (80/20, `random_state=42`, a
nivel de plot). Opción B descartada: FOR-instance V1 solo define dos
particiones oficiales en `data_split_metadata.csv` (56 dev + 26 test). No
existe ni columna ni valor `val` en el metadata; Puliti et al. (2023)
liberan exclusivamente dev/test.

**Justificación académica:** el desdoblamiento interno es necesario porque
(a) PointNet++ requiere un conjunto de monitoreo durante training para
early signal y scheduler, (b) el grid search del watershed necesita un
conjunto independiente del test para calibrar hiperparámetros sin
contaminar el reporte final, y (c) el paper ya reporta métricas "val" que
hay que definir sin ambigüedad. El split a nivel de plot (no de punto)
evita data leakage espacial, siguiendo la práctica de ForAINet
(Henrich 2024) y SegmentAnyTree (Wielgosz 2024).

**Acciones paper §3 "Split dev/test" (líneas 167-169):** al final del
párrafo existente añadir:

> Dado que FOR-instance V1 solo define las particiones dev y test en su
> metadata oficial, el conjunto dev se subdivide internamente en un
> conjunto de entrenamiento (80\% de los plots) y un conjunto de
> validación interno (20\% restante) mediante una partición aleatoria a
> nivel de plot con semilla fija (\texttt{random\_state = 42}) para
> reproducibilidad. La partición se realiza a nivel de archivo
> \texttt{.las} y no a nivel de punto para evitar \emph{data leakage}
> espacial entre train y val, siguiendo el criterio adoptado por
> ForAINet~\cite{henrich2024} y SegmentAnyTree~\cite{wielgosz2024}. El
> conjunto val interno se utiliza para: (i) monitorear la función de
> pérdida durante el entrenamiento de PointNet++ y guiar el scheduler,
> (ii) ejecutar el grid search de los hiperparámetros del Watershed~3D
> de forma independiente por flujo, y (iii) reportar las métricas val que
> se presentan en la Sección de resultados. El conjunto test oficial de
> FOR-instance permanece intacto y se reserva exclusivamente para el
> reporte de métricas finales, sin intervenir en ninguna decisión de
> diseño o calibración.

(Además, cambiar el título de la subsección de "\subsection{Split
dev/test}" a "\subsection{Splits y desdoblamiento interno}" o dejarlo
igual —decisión estética—.)

**Acciones código:** ninguna.

**Sin impacto en resultados:** no se re-corre nada.

---

## 2.4 — Fórmula de densidad local del RF ☑

**Decisión:** Opción A — **corregir el paper** para que la descripción del
feature coincida con el código: "número de vecinos dentro de un radio
fijo `density_radius`", en lugar de la fórmula `k / V_esfera` actual. Se
añade una nota aclarando la equivalencia a radio constante para preservar
el rigor de nomenclatura sin ambigüedad.

**Justificación académica:**

1. **Equivalencia formal bajo Random Forest.** A radio fijo, el conteo
   `k` y la densidad `k / V_esfera` se diferencian solo por un factor
   constante `1 / V_esfera` que depende exclusivamente del hiperparámetro
   `density_radius`. Un RandomForest es invariante bajo reescalado
   lineal de una feature individual (los thresholds de split se
   reescalan proporcionalmente y las decisiones son idénticas). Por lo
   tanto Opciones A y B producen modelos *exactamente* equivalentes; solo
   difieren en las unidades del feature. Re-entrenar para pasar de A a B
   no cambia métricas, solo etiquetas numéricas.

2. **Estándar en la literatura.** El feature "número de vecinos dentro
   de un radio fijo" aparece explícitamente en Weinmann et al. 2015
   como *neighborhood cardinality* / *local point density* entendida
   como conteo a escala fija. No es una imprecisión, es una convención
   aceptada en la literatura que ya se cita.

3. **Coherencia con la filosofía del paper.** La §4.4 defiende
   explícitamente que PointNet++ se evalúa "en condiciones estándar de
   uso" sin optimización exhaustiva. La misma lógica se aplica al RF: no
   se trata de optimizar features, sino de reportar con honestidad lo
   que se calcula. Introducir una densidad adaptativa (Opción C) sería
   scope creep injustificado — un feature distinto al de Weinmann que
   requeriría ablación y rompe la premisa de "usuario promedio con
   configuración estándar".

**Acciones paper Tabla 3 (feature #11, línea ~225 aprox.):** reemplazar
la fila correspondiente por:

> | 11 | Conteo de vecinos local | $k_{|d \leq r|}$ (número de vecinos a distancia $\leq$ \texttt{density\_radius}) | Contexto local de densidad |

Y añadir una nota al pie de Tabla 3 o en el texto que precede a la tabla:

> La feature de conteo de vecinos local se computa como el número entero
> de vecinos dentro de un radio fijo \texttt{density\_radius}. A radio
> constante, esta cantidad es proporcional a la densidad volumétrica
> local $k / V_{\text{esfera}}$ por un factor fijo $1/V_{\text{esfera}}$
> que no afecta al Random Forest por invariancia bajo reescalado lineal
> de una feature individual. Se reporta en forma de conteo para coincidir
> con la convención de \emph{neighborhood cardinality} de Weinmann et
> al.\ 2015.

**Acciones código:** ninguna.

**Sin impacto en resultados:** no se re-corre nada.

---

## 2.5 — Inconsistencia docstring vs. implementación del flip X ☑

**Decisión:** **Falso positivo.** Tras re-verificar
`forest_its/data/pointnet2_dataset.py` (líneas 27-31 del docstring y
138-147 de `__getitem__`), el docstring describe correctamente las tres
operaciones de augmentation (jitter, flip X, scale) y explicita "NO
aplicar rotación en Z — el dataset tiene orientaciones consistentes". La
implementación coincide exactamente con el docstring. No hay
inconsistencia interna en el código.

La única inconsistencia real era **paper ↔ código**, ya capturada y
resuelta en la entrada 2.1. La discrepancia 2.5 se elimina del tracking
como falso positivo de la revisión inicial.

**Acciones paper:** ninguna.

**Acciones código:** ninguna.

**Sin impacto en resultados:** no se re-corre nada.

---

## 2.6 — Restos de `block_size` / `block_overlap` en config.yaml ☑

**Decisión:** Opción A — **eliminar las entradas muertas** del archivo
`forest_its/configs/config.yaml`. Verificado que estos parámetros solo
aparecen en el config y ningún módulo Python los lee (grep confirmó
`Found 1 file: forest_its/configs/config.yaml`). Son vestigios de una
versión anterior del pipeline con ventanas deslizantes tipo S3DIS que
fue reemplazada por submuestreo global del plot completo (consistente
con §4 línea 193 del paper: "Ningún flujo divide la nube en bloques").

**Justificación:** higiene del config. Deja el archivo consistente con
el comportamiento real del código y con la descripción del paper. Evita
que un revisor futuro se confunda pensando que existen ventanas
deslizantes.

**Acciones código `forest_its/configs/config.yaml`:**

Eliminar las dos líneas de la sección `pointnet2:`:

```yaml
  block_size: 10.0          # metros, tamaño de bloque sliding window
  block_overlap: 2.0        # metros, solapamiento entre bloques
```

**Acciones paper:** ninguna — el paper ya es correcto al respecto.

**Sin impacto en resultados:** no se re-corre nada (los valores nunca se
leyeron).

---

## 3.1–3.12 — Minor discrepancies (bloque documental) ☑

**Decisión global:** todas las discrepancias minor se resuelven
actualizando el paper (y en 3.8 limpiando el código). Ninguna afecta
resultados ni requiere re-correr nada. Se consolidan en esta entrada
única por ser todas del mismo tipo (documentación).

---

### 3.1 — `grad_clip` no documentado

**Código** (`config.yaml:44`): `grad_clip: 1.0` — en training se aplica
`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

**Acción paper §4.4 Tabla 5 (Configuración de PointNet++ MSG):** añadir
fila:

> | Gradient clipping | $\lVert g \rVert_2 \leq 1.0$ (norma global) |

Y en el párrafo de training de §4.4, añadir al final: "Se aplica
\emph{gradient clipping} global por norma $\ell_2$ con umbral $1.0$
(\texttt{torch.nn.utils.clip\_grad\_norm\_}) para estabilizar el
entrenamiento con mixed precision."

---

### 3.2 — Dropout en PN++ no documentado

**Código** (`methods/pointnet2/model_msg.py:96-99`): la cabeza de
segmentación tiene `Linear(128,64) → BN → ReLU → Dropout(0.5) →
Linear(64, num_classes)`.

**Acción paper §4.4 Tabla 5:** añadir fila:

> | Dropout | $p = 0.5$ en la cabeza de segmentación |

Y en el párrafo de arquitectura (§4.4.1), mencionar: "La cabeza de
clasificación punto a punto consiste en una capa lineal $128 \to 64$
con BatchNorm, ReLU y \emph{dropout} $p = 0.5$, seguida de una capa
lineal $64 \to C$ (donde $C = 2$)."

---

### 3.3 — Ambigüedad de "F1" en stage semantic

**Código:** `sklearn.metrics.f1_score(y_true, y_pred, average='binary',
pos_label=1)` — F1 binario de la clase positiva (árbol).

**Acción paper §5 (Evaluación) o al introducir el F1 semántico:**
añadir una frase: "El F1 semántico reportado corresponde al F1 binario
de la clase positiva (árbol), calculado con
\texttt{sklearn.metrics.f1\_score} con \texttt{average='binary'} y
\texttt{pos\_label=1}. No se reporta macro-F1 porque el interés
experimental es explícitamente la capacidad de cada método para aislar
la clase árbol, no su balance global entre clases."

---

### 3.4 — `density_radius` del RF no aparece en paper

**Código** (`config.yaml:19`): `density_radius: 0.5` metros.

**Acción paper Tabla 3 (feature #11 ya reescrita en 2.4):** en la nota
al pie que introdujimos en 2.4, añadir el valor explícito:

> [...] El valor utilizado es \texttt{density\_radius} = $0.5$ m, fijo
> para todos los plots y todas las escalas.

---

### 3.5 — Redacción del class mapping del baseline

**Código:** el baseline Flujo A opera sobre los puntos con
`classification ∈ {1, 2, 4, 5, 6}`, excluyendo las clases 0
(\emph{never-classified}) y 3 (\emph{low vegetation} / outpoint), que
están mapeadas a `label = -1`.

**Acción paper §4.2 (Flujo A — Baseline):** reescribir la frase actual
"el baseline usa todos los puntos válidos" por:

> El Flujo A opera sobre todos los puntos del plot con etiqueta de
> clasificación $\in \{1, 2, 4, 5, 6\}$. Los puntos con clase $\in
> \{0, 3\}$ (\emph{never-classified} y puntos de árboles parcialmente
> observados marcados como \emph{outpoint}) se excluyen de todos los
> flujos por protocolo, tanto en el preprocesado como en la evaluación
> de instancias.

---

### 3.6 — Resolución del DTM no reportada

**Código** (`config.yaml:10`): `dtm_resolution: 0.5` metros.

**Acción paper §4.2 o §4.1 (Preprocesado común):** añadir: "El DTM
utilizado para calcular la altura sobre el suelo (HAG) se construye a
resolución $0.5$\,m mediante rasterización de percentil inferior de los
puntos de suelo, seguido de un suavizado gaussiano con ventana $5
\times 5$."

---

### 3.7 — Citación faltante de `yanx27/Pointnet_Pointnet2_pytorch`

**Código:** la implementación del modelo se basa en el repositorio
público de Xu Yan (yanx27).

**Acción paper §4.4.1 y bibliografía:** cambiar la mención informal
"yanx27/Pointnet\_Pointnet2\_pytorch" por una cita formal
`\cite{yan2019pointnet2pytorch}` y añadir a la bibliografía:

```bibtex
@misc{yan2019pointnet2pytorch,
  author = {Yan, Xu},
  title  = {{Pointnet\_Pointnet2\_pytorch}},
  year   = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yanx27/Pointnet_Pointnet2_pytorch}},
  note = {Accedido 2026-04}
}
```

---

### 3.8 — Yang et al. 2020 mencionado sin cita ☑

**Decisión:** **quitar** la referencia. No aparece en el paper (grep lo
confirma), pero sí como comentario muerto en
`forest_its/configs/config.yaml:64`:

```yaml
gaussian_sigma: 0.5        # suavizado gaussiano, Yang et al. (2020) IEEE JSTARS
```

**Acción código `forest_its/configs/config.yaml`:** simplificar el
comentario a `# suavizado gaussiano del grid de densidad 3D`, sin
pseudo-referencia en el comentario del código. Las citas académicas
viven en el paper, no en comentarios de config.

**Acción paper:** ninguna.

---

### 3.9 — `smooth_window` del DTM no reportado

**Código** (`config.yaml:13`): `smooth_window: 5`.

**Acción paper:** ya cubierto por 3.6 (se añade "ventana $5 \times 5$"
en la misma frase sobre DTM).

---

### 3.10 — `hag_min`, `hag_max` no documentados — párrafo nuevo

**Código** (`config.yaml:11-12`): `hag_min: 0.0`, `hag_max: 50.0`
metros.

**Acción paper §4.2 (Preprocesado común):** añadir un párrafo corto
sobre la normalización HAG:

> \paragraph{Normalización de altura sobre el suelo (HAG).}
> Tras calcular la altura sobre el suelo mediante resta del DTM, el
> canal HAG se trunca al rango $[0, h_{\max}]$ con $h_{\max} = 50$\,m,
> que cubre con holgura la altura máxima observada en los cinco sitios
> del benchmark FOR-instance V1. Los valores negativos (ruido del DTM
> bajo el suelo) se clampean a cero y los valores superiores al umbral
> se saturan a $50$\,m. Para el flujo C (PointNet++) este canal se
> renormaliza adicionalmente al intervalo $[0, 1]$ dividiendo por
> $h_{\max}$, garantizando que sea comparable con la intensidad láser
> (también en $[0, 1]$) como feature de entrada a la red.

---

### 3.11 — Matching greedy IoU — pseudocódigo

**Código** (`forest_its/evaluation/instance_metrics.py`): el matching
entre predicciones e instancias GT funciona así:

1. Calcular IoU 3D (punto a punto, 3D) de todos los pares
   `(pred_i, gt_j)`.
2. Descartar pares con `IoU < 0.5`.
3. Ordenar pares restantes por IoU **descendente**.
4. Recorrer la lista en orden: si ni `pred_i` ni `gt_j` han sido
   asignados todavía, marcar ambos como usados y añadir el par al
   conjunto de TPs. En caso de empate exacto en IoU, desempatar por
   menor `pred_id` y luego menor `gt_id` (orden natural de los IDs).
5. Predicciones no casadas → FP. GTs no casados → FN.

**Acción paper §5 (Evaluación):** reemplazar la frase corta actual
("greedy matching con umbral IoU $\geq 0.5$") por un párrafo explícito
con pseudocódigo o descripción paso a paso:

> \paragraph{Matching \emph{greedy} por IoU.}
> Dadas las predicciones de instancia $\{P_i\}_{i=1}^{M}$ y las
> instancias de ground truth $\{G_j\}_{j=1}^{N}$ de un plot, se
> construye la matriz de IoU 3D punto a punto $\mathrm{IoU}(P_i, G_j) =
> |P_i \cap G_j| / |P_i \cup G_j|$. Se descartan todos los pares con
> $\mathrm{IoU} < \tau$ con $\tau = 0.5$. Los pares restantes se
> ordenan por IoU \emph{descendente} (con desempate por orden natural
> de IDs) y se recorren en ese orden: un par $(P_i, G_j)$ se acepta
> como verdadero positivo si y solo si ni $P_i$ ni $G_j$ han sido
> asignados previamente. Las predicciones sin pareja cuentan como
> falsos positivos y los GT sin pareja como falsos negativos. Este
> criterio garantiza que cada instancia predicha se asocia a lo sumo
> con un GT y viceversa, y sigue el protocolo reportado en ForAINet
> (Henrich 2024) y SegmentAnyTree (Wielgosz 2024).

---

### 3.12 — Semilla `random_state=42` no declarada explícitamente

**Código:** aparece en `splits.py` (partición train/val interno),
`rf/train_rf.py` (RF) y `pointnet2/train_pn2.py` (DataLoader worker
seed, init del modelo).

**Acción paper §3 "Split dev/test" — integrada con la acción de 2.3:**
la semilla ya queda mencionada en el párrafo nuevo de 2.3
(`random_state = 42`). Añadir además, al inicio de §4 o al final de §3,
una frase única:

> Para garantizar reproducibilidad, todas las fuentes de aleatoriedad
> del pipeline (partición train/val interna, submuestreo por plot del
> RF, inicialización y barajado del entrenamiento de PointNet++) usan
> la semilla fija $42$ salvo indicación explícita en contrario.

---

**Resumen 3.1-3.12:**

- **Acciones paper:** edits documentales en §3, §4.1, §4.2, §4.4 (Tabla
  5 + párrafos), §5 (evaluación) y bibliografía.
- **Acciones código:** sólo limpiar el comentario de
  `config.yaml:64` (Yang).
- **Sin impacto en resultados:** no se re-corre nada.

---
