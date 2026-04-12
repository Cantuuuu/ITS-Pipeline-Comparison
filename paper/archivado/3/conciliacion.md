# Conciliación paper ↔ código

Decisiones sobre cada discrepancia registrada en `discrepancias.md`.
**No se aplican cambios de código hasta que todas las decisiones estén
tomadas.** Cada entrada queda como `pendiente` hasta que se decide.

Convención por entrada:
- **Discrepancia:** referencia rápida a `discrepancias.md`.
- **Decisión:** `fix code` / `fix paper` / `ambos` / `dejar como está`.
- **Justificación:** por qué.
- **Acción código:** archivos y líneas a tocar (solo registrado, no
  ejecutado todavía).
- **Acción paper:** cambios en `main.tex` (idem).

---

## D1 — Combinación de la semilla por plot del Random Forest

- **Discrepancia:** paper dice "combinación bit a bit" (XOR), docstring
  de `train_rf.py` dice "XOR", código usa **suma** (`+`) con máscara.
- **Decisión:** **fix paper + fix docstring** (opción 1). Mantener el
  código tal cual.
- **Justificación:** suma y XOR son hashes determinísticos equivalentes
  en términos estadísticos; cambiar el código forzaría reentrenar el RF
  y rerunear toda la evaluación sin ganancia. La frase "combinación bit
  a bit" del paper es innecesariamente específica — basta con
  "derivación determinística".
- **Acción código:**
  - `forest_its/methods/rf/train_rf.py:17-19` — reescribir el docstring
    para describir la suma en vez de "XOR".
- **Acción paper:**
  - `paper/main.tex:175` (§3.3) — cambiar "combinación bit a bit de la
    semilla global con el CRC32 del nombre del plot" por una formulación
    neutra (p. ej. "una semilla derivada de forma determinística del
    nombre del plot vía CRC32, de modo que cada plot reciba siempre el
    mismo subconjunto de puntos con independencia del orden de
    iteración").
  - `paper/main.tex:244` (§4.4) — alinear con la nueva redacción de
    §3.3.

---

## D2 — Tie-breaking del greedy matching de instancias

- **Discrepancia:** paper promete desempate por `(pred_id, gt_id)`,
  código usa `np.argsort` (quicksort, sin tie-break secundario).
- **Decisión:** **fix code minimal** (opción 3): añadir `kind="stable"`
  al `argsort`.
- **Justificación:** con stable sort sobre `iou_matrix.ravel()`
  (row-major, índice plano = `pred_idx * n_gt + gt_idx`) y dado que
  `pred_unique` y `gt_unique` están ya ordenados ascendentemente por
  `np.unique`, los empates quedan resueltos naturalmente por
  `(pred_id menor, gt_id menor)` — exactamente lo que promete el paper.
  Cambio de una sola palabra, sin reescritura, y los empates exactos en
  IoU 3D float64 son tan raros que casi con seguridad ningún número de
  evaluación cambia.
- **Acción código:**
  - `forest_its/evaluation/instance_metrics.py:100` — cambiar
    `np.argsort(-iou_matrix.ravel())` por
    `np.argsort(-iou_matrix.ravel(), kind="stable")`.
  - Rerunear `run_*_pipeline.py --stage instance` en val y test para
    confirmar que los CSVs no cambian (o se actualizan si cambian).
- **Acción paper:** ninguna.

---

## D3 — Feature #11 (densidad local) acotada por k vecinos

- **Discrepancia:** paper define cardinalidad de la bola de radio `r`,
  código solo cuenta vecinos dentro del top-k → satura y difiere entre
  k=20 y k=50.
- **Decisión:** **fix code → cardinalidad real con `query_ball_point`,
  feature única compartida entre escalas, total baja a 27 features**
  (opción 1, variante "bajar a 27").
- **Justificación:** la implementación actual deja la feature degradada
  (satura en `k` para zonas densas en una nube de ~18k pts/m²) y
  además contradice la afirmación del paper de que es idéntica en
  ambas escalas. Computar la densidad una vez por punto vía
  `cKDTree.query_ball_point(..., return_length=True)` es barato y
  honra la definición teórica de cardinalidad volumétrica. Bajar a 27
  features (en lugar de duplicar la columna) elimina la redundancia y
  deja al paper describiendo exactamente lo que hace el código.
- **Acción código:**
  - `forest_its/preprocessing/features_rf.py:36-45` — `FEATURE_NAMES_BASE`
    deja de incluir `density`; `FEATURE_NAMES_28` se renombra a
    `FEATURE_NAMES_27` y queda como
    `[f"{n}_k20" for n in BASE_13] + [f"{n}_k50" for n in BASE_13] + ["density"]`,
    donde `BASE_13` son los 13 features actuales menos `density`.
  - `forest_its/preprocessing/features_rf.py:119-157`
    (`compute_local_features`) — eliminar el cómputo de `density` de
    la función local; pasar a producir 5 features locales en lugar de
    6. Actualizar el shape de retorno.
  - `forest_its/preprocessing/features_rf.py:182-207` (`_features_chunk`)
    — pasar de `(N, 28)` a `(N, 27)`. Para los 26 features dependientes
    de escala, seguir usando los slots `[0:13]` y `[13:26]`. Recibir
    `density_per_point` como argumento adicional (precomputado fuera
    del loop) y escribirlo en el slot 26.
  - `forest_its/preprocessing/features_rf.py:210-281`
    (`compute_features_batch`) — antes del particionado en chunks,
    llamar a `tree.query_ball_point(xyz, r=density_radius, return_length=True)`
    una sola vez y pasar el array `(N,)` resultante a `_features_chunk`.
  - `forest_its/preprocessing/features_rf.py:284-343`
    (`compute_features_for_plot`) — actualizar el shape esperado del
    cache (la validación `features.shape[0] == valid_mask.sum()` queda
    igual, pero conviene también verificar `shape[1] == 27` para
    invalidar caches viejos automáticamente).
  - **Invalidar `output/features_cache/`** completo (los caches viejos
    son `(N, 28)` y deben recomputarse).
  - Reentrenar el RF: `python -m forest_its.methods.rf.train_rf`.
  - Rerunear el pipeline RF en val y test:
    `run_rf_pipeline.py --stage semantic` + `--stage instance` (los
    best params del watershed pueden cambiar tras reentrenar, así que
    también `python -m forest_its.evaluation.grid_search --methods rf`).
- **Acción paper:**
  - `paper/main.tex:210-213` (§4.4 "Features") — actualizar el conteo
    "28 features = 14 × 2 escalas" a "27 features = 13 × 2 escalas + 1
    densidad volumétrica compartida".
  - `paper/main.tex:215-238` (Tabla `tab:rf-features`) — actualizar
    título y caption: ahora son 13 features × 2 escalas + 1
    independiente de escala. Mover la fila #11 (densidad) fuera del
    bloque por escala o anotarla como "compartida entre escalas".
    Actualizar la numeración (las features cambian de índice).
  - `paper/main.tex:240-241` (párrafo "Nota sobre la feature #11") —
    reescribir: la feature ahora **es** la cardinalidad real de la
    bola de radio `r` (no un conteo del top-k); el argumento sobre
    invariancia bajo reescalado del RF ya no aplica de la misma forma
    porque el cambio sí elimina el sesgo de saturación.
  - Cualquier otra mención a "28 features" en el manuscrito (revisar
    abstract, introducción, conclusiones) — buscar y actualizar a 27.

---

## D4 — Tipo de filtro de suavizado del DTM

- **Discrepancia:** paper dice "ventana 5×5" sin tipo; código usa
  `uniform_filter` (media), no gaussiano.
- **Decisión:** **fix code → gaussiano** (opción 2). El recomputo no es
  un problema porque aún no hay output ejecutado.
- **Justificación:** un suavizado gaussiano es la convención más común
  en la literatura de DTM forestal y el lector va a leer
  "suavizado de ventana 5×5" como gaussiano por defecto. Como en este
  punto del proyecto no hay HAG cacheado ni features RF cacheados ni
  modelos entrenados, el costo de recomputar es nulo.
- **Acción código:**
  - `forest_its/preprocessing/ground_filter.py:17` — sustituir el
    import de `uniform_filter` por `gaussian_filter`.
  - `forest_its/preprocessing/ground_filter.py:129` — reemplazar
    `uniform_filter(dtm_grid.astype(np.float64), size=smooth_window)`
    por `gaussian_filter(dtm_grid.astype(np.float64), sigma=...)`.
    Subdecisión menor: traducir `smooth_window=5` (kernel 5×5) a un
    `sigma` equivalente. Convención razonable: `sigma = smooth_window / 2`
    → `sigma=2.5`. Mantener `cfg.preprocessing.smooth_window` como
    parámetro pero documentar en config.yaml que se usa como base para
    el sigma del gaussiano.
  - Actualizar el docstring del módulo (`ground_filter.py:1-15`) para
    mencionar gaussiano.
- **Acción paper:**
  - `paper/main.tex:206` — cambiar "suavizado de ventana 5 × 5" por
    "suavizado gaussiano con sigma equivalente a una ventana 5 × 5
    (σ ≈ 2,5 celdas)".

---

## D5 — Construcción del DTM: fallback no documentado + uso de mediana

- **Discrepancia:** paper dice "puntos clasificados como suelo"; código
  usa **mediana** por celda cuando hay clase 2 y cae a percentil 5 sobre
  todos los puntos cuando hay <100 puntos de clase 2.
- **Decisión:** **opción 4 — limpiar el código (eliminar el fallback
  muerto), mantener mediana, documentar en paper**.
- **Justificación:** en FOR-instance todas las colecciones tienen clase
  2 anotada con miles de puntos por plot, así que la rama del fallback
  (`<100` puntos clase 2 → percentil 5 sobre todos los puntos) **nunca
  se activa** en este experimento. Tener código muerto invita a dudas
  del revisor. La mediana sobre puntos clase 2 es razonable porque la
  anotación humana de FOR-instance es de alta calidad y los puntos
  clase 2 son suelo verificado (no hay vegetación rasante contaminando
  como sí pasaría con un filtro morfológico ciego).
- **Acción código:**
  - `forest_its/preprocessing/ground_filter.py:1-15` — actualizar el
    docstring del módulo: eliminar referencia al "Paso 2 (fallback):
    filtro morfológico mínimo" y dejar solo el camino con clase 2 +
    mediana.
  - `forest_its/preprocessing/ground_filter.py:22-52`
    (`extract_dtm` docstring) — eliminar las menciones al fallback y
    al "Paso 2".
  - `forest_its/preprocessing/ground_filter.py:53-63` — eliminar el
    bloque condicional `if terrain_mask.sum() > 100`. Ahora `extract_dtm`
    requiere que `classification` no sea `None` y que `(classification == 2).sum() > 0`.
    Si no se cumple, lanzar `ValueError("extract_dtm requiere puntos
    clasificados como suelo (clase 2 de FOR-instance)")`. Esto convierte
    una rama silenciosa en un error explícito.
  - `forest_its/preprocessing/ground_filter.py:95-103` — eliminar la
    rama `if use_terrain_class: ... else:` y dejar solo
    `dtm_grid[cy, cx] = np.median(cell_z)`.
  - Verificar que ningún test ni script llame a `extract_dtm` sin
    `classification` (búsqueda rápida en la repo).
- **Acción paper:**
  - `paper/main.tex:206` — extender la frase: "El DTM utilizado para
    calcular la HAG se construye por plot tomando la mediana de Z por
    celda sobre los puntos clasificados como suelo (clase 2 de
    FOR-instance), rasterizado a una resolución horizontal de 0,5 m,
    e interpolado con un suavizado gaussiano (σ ≈ 2,5 celdas) para
    reducir artefactos locales." (Compatible con el cambio de D4.)

---

## D6 — `seg F1` reportado vs definido (binario vs weighted)

- **Discrepancia:** paper define `seg F1` como F1 binario `pos_label=1`;
  los pipelines logean `sem_f1_weighted` como "F1 weight". Riesgo de
  llenar la tabla del paper con la columna equivocada.
- **Decisión:** **opción 3 — limpieza fuerte:** renombrar
  `f1_tree` → `seg_f1` en `semantic_metrics.py` (columna CSV pasa a
  `sem_seg_f1`, alineada textualmente con la notación del paper) y
  eliminar `f1_weighted` y `f1_notree` del dict de retorno (y por tanto
  del CSV) junto con su log "F1 weight".
- **Justificación:** la única F1 que el paper define es la binaria de
  la clase árbol; mantener `f1_weighted` y `f1_notree` en el CSV es
  carga conceptual sin uso (no aparecen en ninguna tabla del paper) y
  abre la puerta exactamente al bug que motiva esta entrada (que un
  futuro lector del CSV agarre la columna equivocada). Como no hay
  output ejecutado todavía, no hay CSVs viejos cuyas columnas haya que
  preservar. Renombrar a `seg_f1` (en lugar de dejar `f1_tree`) hace
  que el nombre de la columna empate con la notación del manuscrito y
  quita ambigüedad cuando un revisor cruce código y paper.
- **Acción código:**
  - `forest_its/evaluation/semantic_metrics.py:36-41` (docstring de
    `compute_semantic_metrics`) — actualizar la lista de claves
    retornadas: borrar `f1_weighted` y `f1_notree`, renombrar `f1_tree`
    a `seg_f1`. Añadir una línea explicando que `seg_f1` es el F1
    binario `pos_label=1` y que es el único valor que aparece en las
    tablas del paper.
  - `forest_its/evaluation/semantic_metrics.py:57-59` — eliminar las
    líneas de cómputo de `f1_w` y `f1_notree`. Renombrar `f1_tree` a
    `seg_f1`. Queda solo:
    `seg_f1 = float(f1_score(y_true, y_pred, pos_label=1, labels=[0, 1]))`.
  - `forest_its/evaluation/semantic_metrics.py:61-71` (dict de retorno)
    — eliminar las claves `f1_weighted` y `f1_notree`, renombrar
    `f1_tree` → `seg_f1`.
  - `forest_its/methods/rf/train_rf.py:242` — eliminar el log
    `F1 weight: {sem_metrics['f1_weighted']:.4f}`. Reemplazarlo por
    `seg F1: {sem_metrics['seg_f1']:.4f}` (binario, clase árbol).
  - `forest_its/methods/rf/run_rf_pipeline.py:325` — sustituir
    `df['sem_f1_weighted']` por `df['sem_seg_f1']` y la etiqueta
    `F1 weight:` por `seg F1:`.
  - `forest_its/methods/pointnet2/run_pn2_pipeline.py:345` — mismo
    cambio: `df['sem_f1_weighted']` → `df['sem_seg_f1']`, etiqueta
    `F1 weight:` → `seg F1:`.
  - Verificación: `grep -rn "f1_weighted\|f1_notree\|f1_tree" forest_its/`
    debe quedar vacío después del refactor (excepto cachés `.pyc`).
  - `forest_its/scripts/generate_figures.py` — revisar si lee
    `sem_f1_weighted` o `sem_f1_tree` para alimentar la Tabla 7. Si lo
    hace, ajustar a `sem_seg_f1`. Si todavía no implementa la Tabla 7,
    dejar la nota de que cuando se implemente debe leer `sem_seg_f1`.
- **Acción paper:**
  - `paper/main.tex:379-383` (§5.1) — confirmar que la definición
    queda como "seg F1 = `sklearn.metrics.f1_score(..., pos_label=1,
    average='binary')`". El nombre `seg F1` ya coincide con la columna
    `sem_seg_f1` resultante, así que no hace falta tocar la fórmula
    pero conviene una frase añadida del estilo: "En los CSVs de
    salida esta métrica se serializa como la columna `sem_seg_f1`."

---

## D7 — Coverage = recall en la implementación

- **Discrepancia:** la fórmula del código hace `coverage == recall` por
  construcción; el paper sugiere que podrían diferir.
- **Decisión:** **opción 3 — fix code: coverage genuinamente distinto
  del recall.** Redefinir coverage como "fracción de árboles GT que
  tienen al menos una predicción solapando con IoU ≥ umbral, sin
  imponer matching único".
- **Justificación:** la frase del paper ya insinúa que coverage es
  "recall sin penalización por sobre-segmentación". La definición
  natural de eso es: un GT cuenta como "cubierto" si **alguna** pred
  lo solapa por encima del umbral, aunque esa pred quede unmatched en
  el greedy 1-a-1. Esa métrica sí difiere del recall cuando hay
  sobre-segmentación: dos predicciones que solapan el mismo GT (una
  partida en dos) hacen que recall pierda al GT (la "perdedora" del
  greedy queda como FP, pero el GT quedaría matched solo si hay un
  emparejamiento dominante; en cambio coverage cuenta el GT como
  cubierto siempre que **alguna** pred supere el umbral). Como aún no
  hay output ejecutado el cambio es barato, y deja al paper midiendo
  algo informativamente distinto al recall — justo la motivación
  declarada.
- **Acción código:**
  - `forest_its/evaluation/instance_metrics.py:43-62` (docstring de
    `match_instances`) — actualizar la definición de coverage en la
    sección "Returns": "coverage = fracción de GT trees con ≥1
    predicción cuyo IoU 3D ≥ iou_threshold (sin imponer matching
    único; difiere de recall cuando hay sobre-segmentación)".
  - `forest_its/evaluation/instance_metrics.py:87-93` — la matriz
    `iou_matrix` ya está calculada antes del greedy. Aprovecharla:
    inmediatamente después de su construcción y antes del greedy
    matching, computar
    `covered_gt = int((iou_matrix >= iou_threshold).any(axis=0).sum())`.
    Este es el número de columnas (GTs) que tienen al menos una fila
    (pred) con IoU ≥ umbral.
  - `forest_its/evaluation/instance_metrics.py:118` — sustituir
    `coverage = tp / n_gt` por `coverage = covered_gt / n_gt`.
  - `forest_its/evaluation/instance_metrics.py:71-77` (`empty_result`)
    — `coverage` sigue siendo `0.0` en los casos degenerados (n_gt==0
    o n_pred==0); no hay que cambiar nada ahí.
  - Verificar que ningún test asume `coverage == recall` (búsqueda
    rápida `grep -n "coverage" forest_its/`).
- **Acción paper:**
  - `paper/main.tex:406` (§5.2) — reescribir la frase para hacer
    explícito el matiz: "coverage (cobertura), definido como la
    fracción de árboles GT que reciben **al menos una predicción** con
    IoU 3D ≥ τ, **sin imponer matching único**. A diferencia del
    recall, coverage no penaliza la sobre-segmentación: si dos
    predicciones disjuntas solapan el mismo GT por encima del umbral,
    ambas contribuyen a "cubrir" ese GT aunque solo una pueda quedar
    emparejada en el greedy 1-a-1. Por lo tanto coverage ≥ recall
    siempre, y la brecha (coverage − recall) es un indicador directo
    de fragmentación de copas."

---

## D8 — Comentarios PN++ con RTX 4050 obsoletos

- **Discrepancia:** docstrings de `train_pn2.py` y `predict_pn2.py`
  mencionan "RTX 4050, 6GB"; paper declara M4 Pro.
- **Decisión:** **opción 2 — eliminar las menciones de hardware
  específico de los docstrings** y dejar descripciones agnósticas.
  Las cifras concretas viven en el paper (Tabla 3 / apéndice).
- **Justificación:** los docstrings no son el lugar para benchmarking
  numérico — acoplarlos a una GPU concreta los condena a quedar
  desactualizados con cada cambio de máquina (ya pasó). El paper es la
  fuente única de verdad para hardware, batch size efectivo y tiempos.
  Limpieza barata y sin riesgo.
- **Acción código:**
  - `forest_its/methods/pointnet2/train_pn2.py:15` — eliminar la línea
    `"- Verifica que cabe en VRAM (RTX 4050, 6GB)"` de la lista del
    docstring `--dry-run`. Reemplazarla por
    `"- Verifica que el modelo cabe en memoria del dispositivo activo"`
    (agnóstica a GPU/MPS/CPU).
  - `forest_its/methods/pointnet2/predict_pn2.py:23-27` — eliminar el
    bloque "Cobertura esperada con n_passes = N / num_points * 3 …
    Tiempo estimado: ≈ 4.5 min para 5 plots val en RTX 4050." Esta es
    además la cifra de cobertura mal calculada de D9, así que conviene
    borrar todo el bloque y dejar que D9 lo reescriba sin referencia
    a hardware concreto.
- **Acción paper:** ninguna. Las celdas de hardware del apéndice
  (Tabla 8) se completan en D10.

---

## D9 — Cobertura 86% en docstring vs ~95% real

- **Discrepancia:** docstring de `predict_pn2.py` calcula 86%, el cálculo
  teórico da ~95% antes de KNN y >99% después (consistente con el paper).
- **Decisión:** **opción 1 — reponer un bloque corto con el cálculo
  correcto** en el docstring de `predict_pn2.py`, sin mención de
  hardware ni tiempos. El bloque defectuoso ya fue marcado para borrado
  en D8.
- **Justificación:** el `86%` es un error de cálculo. La probabilidad
  de que un punto **no** sea muestreado en una pasada de tamaño `N_s`
  sobre una nube de tamaño `N` es `(1 − 1/N)^{N_s} ≈ exp(−N_s/N)`. Con
  `n_passes = 3·N / N_s`, la probabilidad acumulada de no ser
  muestreado en ninguna pasada es
  `(1 − N_s/N)^{n_passes} ≈ exp(−3) ≈ 4,98%`, lo que da cobertura ≈
  **95%** por punto antes del KNN final. La interpolación KNN sobre
  los puntos no cubiertos lleva la cobertura efectiva a >99%
  (consistente con `paper/main.tex:313`). Cuatro líneas dentro del
  docstring evitan que un futuro lector tenga que rederivar el cálculo
  desde el paper.
- **Acción código:**
  - `forest_its/methods/pointnet2/predict_pn2.py:23-27` — el bloque
    "Cobertura esperada con n_passes = N / num_points * 3: …
    Tiempo estimado: ≈ 4.5 min para 5 plots val en RTX 4050." se borra
    (acción D8) y se reemplaza por:
    ```
    Cobertura teórica con n_passes = clip(3·N / N_s, 50, 1000):
      P(punto no muestreado en una pasada) ≈ exp(-N_s/N)
      P(punto no muestreado tras n_passes) ≈ exp(-3) ≈ 5%
      → cobertura por punto ≈ 95% antes del KNN final.
      La interpolación KNN sobre puntos no cubiertos lleva la
      cobertura efectiva a >99% en todos los plots observados.
    ```
    Sin cifras de hardware ni de tiempo (esas viven en el paper §4.4
    y en el apéndice Tabla 8).
- **Acción paper:** ninguna. `paper/main.tex:313` ya reporta ">99%"
  correctamente.

---

## D10 — Apéndice tabla 8 con celdas `[COMPLETAR]`

- **Discrepancia:** filas "Hardware PointNet++" y "Código" del apéndice
  todavía pendientes.
- **Decisión:** **opción 1 — completar parcialmente.** Llenar lo que
  ya sabemos del config (hardware + batch + epochs + precisión), dejar
  el tiempo total como `[pendiente medir]` hasta que se ejecute el
  pipeline definitivo en M4 Pro, y mantener la URL del repositorio
  como `[PENDIENTE: URL repositorio]` (decisión de publicación).
- **Justificación:** los datos extraíbles del `config.yaml`
  (`pointnet2.epochs=100`, `batch_size=16`, `mixed_precision=bf16`) y
  del paper (`Apple M4 Pro, memoria unificada`) son verdades estables
  que no dependen de ejecutar nada. El tiempo total sí depende de
  correr el pipeline definitivo, así que merece quedar como un
  placeholder visible y específico (en lugar de un genérico
  `[COMPLETAR]`). La URL queda como decisión separada (publicación) y
  se completa el día que se haga público el repo. Honesto sobre lo
  que falta, no sobre-compromete.
- **Acción código:** ninguna.
- **Acción paper:**
  - `paper/main.tex:739` — sustituir
    `Hardware PointNet++  & [COMPLETAR: GPU, memoria, tiempo total]`
    por
    `Hardware PointNet++  & Apple M4 Pro (memoria unificada), batch=16, bf16, 100 epochs; tiempo total: [pendiente medir] \\`.
  - `paper/main.tex:743` — dejar
    `Código & [PENDIENTE: URL repositorio]` tal cual (TODO de
    publicación, no técnico).
  - Antes de la submisión: reemplazar `[pendiente medir]` con la
    cifra real medida en M4 Pro (cuando el pipeline corra).

---

## D11 — Rango de densidades por sitio sin completar

- **Discrepancia:** §3.1 y §6.1 del paper tienen `[COMPLETAR]` en el
  rango de densidades.
- **Decisión:** **opción 3 — medir empíricamente sobre los LAS y citar
  Puliti et al. 2023.** Calcular densidad por plot agregada por
  colección (mín / mediana / máx), reportar el rango global de las
  cinco colecciones (NIBIO, CULS, TUWIEN, RMIT, SCION) en el
  manuscrito y mantener la cita a Puliti como fuente metodológica.
- **Justificación:** lo correcto científicamente es reportar la
  densidad de los datos efectivamente usados (no la del paper original
  que pudo incluir colecciones distintas). El cómputo es trivial y el
  código ya tiene la lógica: `forest_its/scripts/explore_dataset.py:111-116`
  ya calcula `density = n_points / (x_span * y_span)` por plot. Solo
  hace falta extender el script para agregar por colección y
  reportar mín/mediana/máx por sitio + el rango global. Costo: ~20
  líneas y un único pase sobre los LAS.
- **Acción código:**
  - `forest_its/scripts/explore_dataset.py` — añadir una función
    `summarize_density_by_collection(dataset_root)` que itere todos
    los plots, calcule `density = n_points / (x_span * y_span)`
    (ya existe la fórmula en línea 111-116), agrupe por colección
    inferida del nombre del plot/path, y reporte
    `{collection: (min, median, max, n_plots)}`. Imprimir además
    el rango global `(min(all), max(all))` para usarlo directamente
    en el paper.
  - Llamar a la nueva función desde `main()` del mismo script (o
    detrás de un flag `--density-summary`) para no romper el output
    actual del modo exploración.
  - Ejecutar `python -m forest_its.scripts.explore_dataset --density-summary`
    una vez sobre `cfg.paths.dataset_root` y guardar el output como
    referencia (puede ir a `output/dataset_density_summary.txt` o
    quedarse en stdout).
- **Acción paper:**
  - `paper/main.tex:141` (§3.1) — sustituir
    `densidades que van de [COMPLETAR: rango por sitio]~pts/m$^2$`
    por una formulación con cifras reales medidas, p. ej.
    `densidades que van de {min}--{max}~pts/m$^2$ seg\'un la
    colecci\'on (Tabla~\ref{tab:collections}; medido sobre las cinco
    colecciones empleadas, consistente con los rangos reportados por
    Puliti et al.~\cite{puliti2023})`. Sustituir `{min}` y `{max}`
    por los valores que arroje el script.
  - Opcionalmente añadir una columna "Densidad media (pts/m²)" a la
    Tabla~\ref{tab:collections} con la mediana por colección.
  - `paper/main.tex:598` (§6.1, limitación 4) — sustituir
    `densidades de [COMPLETAR]~pts/m$^2$` por
    `densidades de {min}--{max}~pts/m$^2$` (mismos valores).
  - Eliminar los comentarios hardcoded de densidad en el código que
    citan números obsoletos: `forest_its/preprocessing/features_rf.py:22`
    ("RMIT ~454 pts/m²") y
    `forest_its/segmentation/watershed3d.py:9` ("FOR-instance ~18,236
    pts/m²"); reemplazar por una referencia genérica al rango medido
    o eliminar la cifra y citar §3.1 del paper.

---

## D12 — Naming `pn2` vs `pointnet2` en CLI/CSVs

- **Discrepancia:** carpeta `pointnet2/` pero CLIs y CSVs usan `pn2`;
  `CLAUDE.md` ejemplifica con `pointnet2` (que no existe como archivo).
- **Decisión:** **opción 2 — unificar todo a `pointnet2`.** Renombrar
  los archivos del módulo, los argumentos de CLI, los subdirs de
  predicciones y los nombres de CSV para que coincidan con la carpeta
  `forest_its/methods/pointnet2/` y con `cfg.pointnet2.*`.
- **Justificación:** la carpeta y la sección de config ya son
  `pointnet2`, así que el sesgo del codebase apunta al nombre largo.
  `pointnet2` es además auto-explicativo en revisión externa
  (`pn2` se lee como "P número 2" y confunde). Como aún no hay output
  ejecutado, no hay CSVs ni `output/predictions/` viejos que
  renombrar — solo el código fuente. Después del refactor `CLAUDE.md`
  queda correcto sin tener que tocarlo.
- **Acción código:**
  - Renombrar archivos:
    - `forest_its/methods/pointnet2/train_pn2.py` →
      `train_pointnet2.py`
    - `forest_its/methods/pointnet2/predict_pn2.py` →
      `predict_pointnet2.py`
    - `forest_its/methods/pointnet2/run_pn2_pipeline.py` →
      `run_pointnet2_pipeline.py`
  - Renombrar funciones públicas dentro de los archivos:
    - `predict_plot_pn2` → `predict_plot_pointnet2`
    - `run_pn2_pipeline` → `run_pointnet2_pipeline`
  - Actualizar todos los imports cruzados:
    - `forest_its/methods/pointnet2/run_pointnet2_pipeline.py:41` →
      `from forest_its.methods.pointnet2.predict_pointnet2 import predict_plot_pointnet2, load_model`
    - `forest_its/methods/pointnet2/diag_coverage_plot3.py:20` →
      `from forest_its.methods.pointnet2.predict_pointnet2 import load_model`
    - Cualquier otro `from forest_its.methods.pointnet2.predict_pn2`
      / `train_pn2` / `run_pn2_pipeline` (`grep -rn` para encontrar).
  - `forest_its/evaluation/grid_search.py:285` y `argparse choices`
    — sustituir `"pn2"` por `"pointnet2"` en `--methods` (default
    `["baseline", "rf", "pointnet2"]`, choices idem).
  - `forest_its/evaluation/grid_search.py:106` (mensaje de error que
    sugiere `python -m forest_its.methods.{method}.run_{method}_pipeline`)
    — verificar que `f"run_{method}_pipeline"` queda como
    `run_pointnet2_pipeline` cuando `method == "pointnet2"`. Funciona
    automáticamente si el archivo está renombrado.
  - `forest_its/methods/pointnet2/run_pointnet2_pipeline.py:7-8` y
    docstring (líneas 18-22) — sustituir todas las menciones a
    `output/predictions/pn2/`, `train_pn2`, `run_pn2_pipeline`, y
    `--methods pn2` por sus versiones `pointnet2`.
  - `forest_its/methods/pointnet2/run_pointnet2_pipeline.py:85, 123`
    — sustituir `output_dir / "predictions" / "pn2"` por
    `output_dir / "predictions" / "pointnet2"`.
  - `forest_its/methods/pointnet2/run_pointnet2_pipeline.py:239-240`
    — sustituir el nombre del logger `f"pn2_pipeline_{stage}"` y el
    archivo de log `f"pn2_pipeline_{stage}_{split}.log"` por
    `pointnet2_pipeline_*`.
  - `forest_its/methods/pointnet2/run_pointnet2_pipeline.py:288, 328`
    — sustituir los nombres de CSV `pn2_semantic_{split}.csv` y
    `pn2_metrics_{split}.csv` por `pointnet2_semantic_{split}.csv` y
    `pointnet2_metrics_{split}.csv`.
  - `forest_its/methods/pointnet2/run_pointnet2_pipeline.py:298` —
    sustituir `load_best_watershed_params("pn2", cfg)` por
    `load_best_watershed_params("pointnet2", cfg)`.
  - Buscar y eliminar cualquier referencia residual a `pn2`:
    `grep -rn "pn2" forest_its/` debe quedar vacío después del
    refactor (ignorando `__pycache__/` y comentarios históricos
    intencionales).
  - **No hace falta** invalidar cachés ni borrar `output/`: como no
    hay output ejecutado, los nombres viejos `output/predictions/pn2/`
    y `pn2_metrics_*.csv` no existen en disco.
- **Acción paper:** ninguna. El paper ya usa "PointNet++ MSG" /
  "Flujo C" y no menciona ni `pn2` ni `pointnet2` como nombre interno.
