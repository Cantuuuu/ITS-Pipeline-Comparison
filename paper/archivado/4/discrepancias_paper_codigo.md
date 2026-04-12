# Discrepancias paper ↔ código

Revisión sistemática de `paper/main.tex` contra el código en `forest_its/`. Cada
ítem lleva checkbox para conciliación posterior. Las opciones incluyen mi
recomendación marcada con **[REC]**.

Convención: cuando la conciliación afecta al paper, "editar §X.Y". Cuando afecta
al código, nombre del archivo y función.

---

## 1. Dataset y splits

### 1.1 Exclusión explícita de NIBIO2

- [ ] **Paper (§3.1)** dice: "Se excluye **explícitamente** NIBIO2 por una razón metodológica".
- **Código** (`forest_its/data/dataset.py::load_splits`): filtra archivos que
  no existen en disco (`file_path.exists()`), con el comentario: "Filtra
  automáticamente archivos que no existen en disco (ej. NIBIO2 referenciado
  en el CSV pero no descargado)". No hay filtro programático por nombre
  de colección.
- **Impacto**: si alguien descarga NIBIO2 al `FORinstance_dataset/`, el
  pipeline lo incluirá sin aviso. La exclusión no está "explícita" en código.
- **Opciones de conciliación**:
  - [x] **[REC]** Añadir a `load_splits` un `EXCLUDE_FOLDERS = {"NIBIO2"}`
        (o similar) y saltar esas filas del CSV con log explícito. Es una
        línea y deja el código honesto respecto al texto.
  - [ ] Ajustar el paper a "se excluye por no descargar NIBIO2" (menos
        limpio, contradice "explícita").

### 1.2 Conteo de plots/árboles de la Tabla 1

- [ ] **Paper (Tabla 1)** reporta test: NIBIO 6/161, CULS 1/20, TUWIEN 1/35,
      RMIT 1/64, SCION 2/43, total 11/323.
- **Código**: no hay verificación de estos números; `load_splits` lee el CSV
  oficial y confía. Los 32 plots del rango de densidades mencionados en §3.1
  tampoco se verifican.
- **Opciones**:
  - [x] **[REC]** Añadir un script `forest_its/scripts/verify_splits.py`
        que recorra los splits tras `load_splits`, imprima el conteo por
        colección y lo compare contra valores esperados; lanzar aviso si
        difiere. No altera el pipeline, sirve como auditoría al redactar
        tablas finales.
  - [ ] Dejar solo nota en paper: "valores verificados manualmente al
        construir la Tabla 1".

### 1.3 Rango de densidades 454–10 156 pts/m²

- [ ] **Paper (§3.1)** reporta rango y medianas por colección (RMIT 473,
      TUWIEN 1 406, CULS 2 510, SCION 3 064, NIBIO 7 246).
- **Código**: no hay script que calcule densidad por plot/colección. El
  reporte `explore_dataset.py` existe (según `git status`) pero no lo he
  leído — puede o no reportarlo.
- **Opciones**:
  - [x] **[REC]** Verificar que `forest_its/scripts/explore_dataset.py`
        efectivamente computa densidades. Si no, añadir la medición y
        exportar un CSV citable desde el paper.
  - [ ] Dejar los números del paper como "medidos una sola vez, no
        reproducidos desde código".

---

## 2. Preprocesado HAG

### 2.1 σ del suavizado gaussiano del DTM

- [ ] **Paper (§4.2)**: "σ equivale a una ventana 5×5 (σ ≈ 2,5 celdas)".
- **Código** (`preprocessing/ground_filter.py::extract_dtm`): usa
  `σ = smooth_window / 2.0 = 5/2 = 2.5`. ✅ **Consistente**, no requiere
  acción — lo dejo listado para trazabilidad.

### 2.2 Interpolación de celdas vacías del DTM

- [ ] **Paper (§4.2)**: describe mediana por celda + suavizado gaussiano,
      pero **no menciona** cómo se rellenan las celdas sin puntos de clase 2.
- **Código**: `griddata(method='linear')` con fallback `'nearest'` para
  bordes (`ground_filter.py` líneas 111–130).
- **Impacto**: omisión menor en reproducibilidad.
- **Opciones**:
  - [x] **[REC]** Añadir una frase en §4.2 tras "mediana de z por celda":
        "Las celdas sin puntos de suelo se rellenan por interpolación bilineal
        (scipy.interpolate.griddata, método 'linear'), con fallback 'nearest'
        para bordes."
  - [ ] No mencionarlo (detalle de implementación que no cambia resultados).

### 2.3 HAG sobre nube completa vs. nube filtrada (flujo A)

- [ ] **Paper**: no distingue. Implicación del texto es que HAG se calcula
      idéntico en los tres flujos.
- **Código**:
  - Flujo A (`methods/baseline/run_baseline.py::run_baseline_single`)
    computa HAG sobre `xyz_valid` (excluye clases 0 y 3 **antes** de
    calcular el DTM y el HAG).
  - Flujos B y C (`run_rf_pipeline.py`, `run_pointnet2_pipeline.py`)
    computan HAG sobre la nube completa vía `process_plot(data, ...)`.
- **Impacto**: el _extent_ de la grilla del DTM puede diferir unos pocos
  píxeles en los bordes si las clases 0/3 caen en el perímetro del plot.
  El DTM interior es idéntico (los puntos clase 2 son los mismos). Es
  una diferencia marginal pero introduce asimetría entre flujos sin
  justificación.
- **Opciones**:
  - [x] **[REC]** Armonizar: en `run_baseline.py::run_baseline_single`
        calcular HAG con `compute_hag(data["xyz"], classification=data["classification"], ...)`
        (nube completa) y luego indexar con `valid_mask`. Elimina la
        asimetría con una línea cambiada.
  - [ ] Dejar el código y añadir nota al paper aclarando que la
        diferencia es despreciable por usar la misma clase 2.

---

## 3. Features del Random Forest

### 3.1 Convención de eigenvalores: raw vs. normalizados

- [ ] **Paper (Tabla 2)** define las features con λ1, λ2, λ3 sin aclarar
      si son autovalores crudos o normalizados (λi / Σλ).
- **Código** (`preprocessing/features_rf.py::compute_eigenfeatures`):
  - Linealidad, planaridad, esfericidad, anisotropía, cambio de curvatura →
    usan eigenvalores **crudos**. Como son ratios homogéneos en λ, el
    resultado es idéntico a la versión normalizada.
  - **Omnivarianza** → usa `(lam1 * lam2 * lam3) ** (1/3)` **crudo**.
    Con la convención Weinmann (normalizada) el valor es
    `omni_raw / (Σλ)`, es decir distinto y de menor magnitud. La
    convención estándar de Weinmann 2013/2014/2015 es **normalizada**.
  - **Eigenentropía** → usa `-Σ l_i_norm * log(l_i_norm)` con valores
    **normalizados** (consistente con Weinmann).
  - **Suma de eigenvalores** → crudos, con comentario explícito
    "CRUDOS, no normalizados".
  - Docstring del módulo dice: "Features 1-6 y 8 usan eigenvalores
    normalizados" — **incorrecto respecto al propio código** (feat. 4
    omnivarianza no está normalizada).
- **Impacto**: la feature _omnivariance_k20_ y _omnivariance_k50_ no
  siguen Weinmann. Afecta magnitud absoluta pero no rankings — el RF
  usa splits, por lo que el impacto práctico es nulo. Sin embargo es
  una incoherencia que un revisor señalará si alguien reproduce el
  experimento con una implementación canónica.
- **Opciones**:
  - [x] **[REC]** Cambiar el código para usar normalizados en
        omnivarianza: `(l1n * l2n * l3n) ** (1/3)`, y corregir el docstring
        del módulo (`features_rf.py` líneas 17–19) para reflejar la realidad.
        Recalcular cache de features y re-entrenar RF.
  - [ ] Ajustar paper y docstring para que declaren explícitamente la
        convención **raw** para omnivarianza. Justificar que el RF es
        invariante a transformaciones monótonas por split. Más corto pero
        contradice la literatura citada.

### 3.2 Convención de eigenentropía en el paper

- [ ] **Paper (Tabla 2)** escribe: `-Σ λi ln λi` (sin normalizar).
- **Código**: usa eigenvalores normalizados (correcto por Weinmann).
- **Impacto**: el paper está matemáticamente impreciso si se lee literal —
  la entropía de Shannon requiere una distribución de probabilidad
  (Σ pi = 1), por lo que los λi deben estar normalizados. El código es
  correcto; el paper hay que corregirlo.
- **Opciones**:
  - [x] **[REC]** Editar Tabla 2, fila 6: cambiar la fórmula a
        `-Σ (λi/Σλ) ln(λi/Σλ)` o introducir nota al pie:
        "λi denotan los autovalores normalizados λi/(λ1+λ2+λ3) siguiendo
        Weinmann et al. (2015) salvo en la fila 7 (Suma de eigenvalores)
        que usa eigenvalores crudos".
  - [ ] No tocar el paper y argumentar que es "notación laxa" — no lo
        recomiendo, un revisor ISPRS pediría el cambio.

### 3.3 Divisor de la matriz de covarianza: k vs. k-1

- [ ] **Paper**: no lo menciona.
- **Código** (`compute_eigenfeatures`): divide por `k` con comentario
  explícito ("consistente con Weinmann et al. 2014"). Las funciones
  derivadas (ratios) son invariantes al divisor, pero `sum_eigenvalues`
  y `omnivariance` (si se dejara cruda) sí dependen de k vs. k-1.
- **Opciones**:
  - [ ] **[REC]** Añadir una frase corta en §4.3 tras "matriz de
        covarianza local" — "computada con divisor k siguiendo Weinmann
        et al. (2014)". Una línea, elimina ambigüedad.
  - [ ] Ignorar por ser detalle sin efecto perceptible.

### 3.4 `max_neighbor_distance = 5.0 m` (cap KNN) no en el paper

- [ ] **Paper**: menciona k=20 y k=50 vecinos, sin tope de distancia.
- **Código** (`features_rf.py::_features_chunk`): si el k-ésimo vecino
  está a más de `max_neighbor_distance = 5.0 m`, se descartan los
  vecinos que violen ese radio. Si quedan <4 vecinos válidos, la
  feature queda en ceros.
- **Impacto**: en plots de baja densidad (RMIT, mediana 473 pts/m²),
  el 50-ésimo vecino puede exceder 5 m y la feature se degrada. Esto
  **sí** afecta el RF y debería estar documentado.
- **Opciones**:
  - [x] **[REC]** Documentar en §4.3 tras "vecindades (k=20 y k=50)":
        "Para evitar que en plots de baja densidad los vecinos queden
        demasiado alejados, se impone un radio máximo de 5,0 m; los
        vecinos a mayor distancia se descartan y los puntos con menos
        de 4 vecinos válidos reciben features en cero." También reportar
        qué fracción de puntos cae en ese régimen (seguramente <1% pero
        hay que medirlo).
  - [ ] Eliminar el cap del código (no recomendado — fue puesto por
        una razón práctica).

### 3.5 Intensidad normalizada "por plot"

- [ ] **Paper (Tabla 2, fila 13)**: "I / I_max por plot".
- **Código** (`data/dataset.py::load_las`): `imax = raw_intensity.max()`
  sobre los puntos del plot cargado. ✅ Consistente.

### 3.6 Submuestreo balanceado 12 000 pts/clase/plot

- [ ] **Paper (§4.3)**: "hasta 12 000 puntos por clase en cada plot".
- **Código** (`train_rf.py`): aplica el cap a cada clase por separado
  dentro de cada plot, con semilla `(random_state + crc32(stem)) & 0x7FFFFFFF`.
  ✅ Consistente con el texto.

---

## 4. PointNet++

### 4.1 Radios y npoints de las capas Set Abstraction no están en el paper

- [ ] **Paper (§4.4)**: "cuatro módulos de agrupamiento jerárquico (Set
      Abstraction)" sin detallar radios, npoints ni nsamples.
- **Código** (`methods/pointnet2/model_msg.py`):
  ```
  SA1: npoint=1024, radii=[0.1, 0.2] m, nsamples=[16, 32]
  SA2: npoint=256,  radii=[0.2, 0.4] m, nsamples=[16, 32]
  SA3: npoint=64,   radii=[0.4, 0.8] m, nsamples=[16, 32]
  SA4: npoint=16,   radii=[0.8, 1.6] m, nsamples=[16, 32]
  FP4: mlp=[512, 512]
  FP3: mlp=[512, 256]
  FP2: mlp=[256, 128]
  FP1: mlp=[128, 128]
  ```
- **Impacto**: un revisor que quiera reproducir no puede hacerlo sólo
  con el paper; tiene que mirar yanx27. Nota: los radios están en **espacio
  normalizado por plot** (recordar que el sample está dividido por el
  radio máximo), no en metros reales — el comentario del código
  ("metros") es incorrecto y debe aclararse.
- **Opciones**:
  - [ ] **[REC]** Añadir una tabla corta en §4.4 con los valores exactos
        de SA/FP (npoint, radii, nsamples, mlp). Aclarar que los radios son
        en el **espacio normalizado** del sample (post división por radio
        máximo), no metros absolutos. Corregir el comentario del código
        en `model_msg.py` líneas 61–83.
  - [x] Mantener la referencia al repo yanx27 como es y añadir sólo la
        aclaración del espacio normalizado.

### 4.2 Pre-centrado XY a nivel de plot (antes de normalizar por sample)

- [ ] **Paper (§4.4)**: menciona "normalización per-sample por centroide
      y radio máximo" pero NO menciona el pre-centrado a nivel de plot.
- **Código**:
  - `data/pointnet2_dataset.py::ForInstanceDataset.__init__` pre-centra
    XY sobre el centroide del plot antes de almacenar.
  - `predict_pointnet2.py::predict_plot_pointnet2` replica este
    pre-centrado (líneas 153–163).
- **Impacto**: el pre-centrado reduce error numérico de float32 pero
  no cambia el resultado matemático porque la normalización per-sample
  posterior lo absorbe. Aun así, debería documentarse por transparencia.
- **Opciones**:
  - [x] **[REC]** Añadir media frase en §4.4 ("Entrenamiento" o
        "Inferencia"): "Previo a la normalización per-sample, las
        coordenadas XY se centran en el centroide del plot para reducir
        error numérico en float32".
  - [ ] Ignorar (no afecta resultados).

### 4.3 Batch size de inferencia (16 subsamples por forward)

- [ ] **Paper**: no lo menciona (razonable, es detalle de implementación).
- **Código** (`predict_pointnet2.py`): `_INFER_BATCH = 16`.
- **Opciones**:
  - [x] **[REC]** No añadir al paper. No requiere acción.

### 4.4 n_passes fórmula

- [ ] **Paper (§4.4, ec. 1)**: `n_passes = min(1000, max(50, ⌊3N/Ns⌋))`.
- **Código**: `n_passes = min(1000, max(50, int(N / num_points * 3)))`.
  ✅ `int()` de un positivo equivale a floor. Consistente.

### 4.5 Interpolación KNN k=3 con pesos 1/d

- [ ] **Paper (§4.4)**: "interpolación KNN ponderada por el inverso
      de la distancia (wj ∝ 1/dj) con k=3".
- **Código**: ✅ consistente (`assign_uncovered_points` con k=3 y pesos 1/d).

### 4.6 Mixed precision bfloat16 sin GradScaler

- [ ] **Paper (§4.4)**: "mixed precision (torch.amp, bfloat16)".
- **Código**: ✅ consistente (bf16, sin GradScaler — comentario explícito).

### 4.7 Pesos de clase NLLLoss: "frecuencia inversa"

- [ ] **Paper (§4.4)**: "NLLLoss ponderada por la frecuencia inversa de
      cada clase".
- **Código** (`train_pointnet2.py::compute_class_weights`):
  `weights = total / (2 * counts + 1e-6)`. Es proporcional a 1/count y
  normalizado de modo que la suma dé `total/counts.sum() = 1` (bueno,
  cerca de N). Matemáticamente cuenta como "frecuencia inversa" con un
  factor 1/2 que compensa dos clases.
  ✅ Consistente.

### 4.8 `num_workers`, `persistent_workers` en DataLoader

- [ ] **Paper**: no lo menciona.
- **Código**: `num_workers=6` (train) y `num_workers_val=4`.
- **Opciones**: no añadir al paper. Detalle de implementación irrelevante
  para el experimento.

---

## 5. Watershed 3D

### 5.1 Paso 1 — voxelización en (X, Y, HAG)

- [ ] **Paper (§4.5)**: "Paso 1 — Voxelización en (X, Y, HAG)".
- **Código** (`watershed3d.py`): el módulo recibe `points: (N, 3)` y no
  sabe que la tercera coordenada es HAG — son los pipelines
  (`run_baseline.py`, `run_rf_pipeline.py`, `run_pointnet2_pipeline.py`)
  los que construyen `np.column_stack([x, y, hag])` antes de llamar a
  `watershed3d()`. ✅ Consistente, pero la doc del watershed no lo
  dice explícitamente.
- **Opciones**:
  - [x] **[REC]** Actualizar el docstring de
        `watershed3d.py::watershed3d` para decir "Se asume que la tercera
        coordenada del input es HAG, no Z absoluto". Una línea. No afecta
        al paper.
  - [ ] Dejar como está; los pipelines ya lo documentan.

### 5.2 Umbral de altura: `threshold_abs` + filtro por top_z

- [ ] **Paper (§4.5)**: "umbral absoluto igual a min_tree_height".
- **Código**: primero `peak_local_max(..., threshold_abs=min_tree_height)`
  sobre el CHM en metros (✅) y **adicionalmente** un segundo filtro
  `top_z >= min_height_voxels` después. El segundo filtro es redundante
  dado el primero, pero correcto.
- **Opciones**:
  - [x] **[REC]** Eliminar el segundo filtro redundante en
        `watershed3d.py` líneas 179–186 por limpieza. No afecta resultados.
  - [ ] Dejar como está (redundante pero inocuo).

### 5.3 `min_crown_radius_m` como separación mínima entre picos

- [ ] **Paper (§4.5)**: "separación mínima entre picos dada por
      min_crown_radius_m / voxel_size".
- **Código**: `min_distance_voxels = max(1, int(min_crown_radius_m / voxel_size))`.
  ✅ Consistente (el `max(1, ...)` es guardia numérica).

### 5.4 Semillas: comentarios del código citan literatura distinta a la del paper

- [ ] **Paper (§4.5)**: cita `dalponte2016` para la práctica estándar de
      seeding con envolventes tipo CHM.
- **Código** (`watershed3d.py`, líneas 29–46): cita "Li 2012, Chen 2006,
  Dalponte-Coomes 2016" en el docstring. Y comentarios puntuales citan
  "Yang et al. 2020 IEEE JSTARS" para σ=0.5 y "Chen et al. (2022)" para
  el ajuste empírico de `min_crown_radius_m`.
- **Impacto**: el paper no tiene esas referencias en su bibliografía.
  Si el trabajo cita a Li/Chen/Yang en comentarios, conviene que el
  paper también, o borrarlas del código para no prometer algo que no
  está justificado en el manuscrito.
- **Opciones**:
  - [ ] **[REC]** Decidir un único conjunto de referencias y mantenerlo
        tanto en paper como en código:
    - Opción A: añadir Li 2012, Chen 2006 y Yang 2020 al `thebibliography`
      del paper (sec 2.2) y citarlas en §4.5 donde corresponda.
    - [x] Opción B: retirar esas citas de los comentarios del código y
          dejar sólo Dalponte-Coomes 2016 (ya en paper). Más rápido.
    - Yo iría con **B** si no se quiere expandir la sección 2.2.
  - [ ] Mantener las dos bibliografías divergentes (no recomendado —
        crea ruido en revisión).

---

## 6. Métricas y evaluación

### 6.1 Definiciones de TP, FP, FN

- [ ] **Paper (§5.2)** y **código** (`evaluation/instance_metrics.py`) ✅
      consistentes: greedy matching por IoU descendente, τ=0.5, desempate
      estable por (pred_idx, gt_idx) ascendente vía stable sort sobre índice
      plano `pred_idx * n_gt + gt_idx`.
- **Verificación**: `np.unique` devuelve valores ordenados, así que
  pred_unique y gt_unique están en orden ascendente de id y el tiebreak
  efectivamente es "primero menor pred_id, luego menor gt_id". ✅

### 6.2 Coverage

- [ ] **Paper (§5.2)** y **código** ✅ consistentes: `covered_gt` se
      computa **antes** del greedy, contando cuántos GT tienen al menos
      una predicción por encima del umbral. Coverage ≥ Recall por
      construcción.

### 6.3 `seg F1` binario con `pos_label=1`

- [ ] **Paper (§5.1)**: "sklearn.metrics.f1_score con average='binary'
      y pos_label=1".
- **Código** (`semantic_metrics.py`): `f1_score(y_true, y_pred, pos_label=1, labels=[0, 1])`.
  El parámetro `average` no se pasa explícitamente y sklearn cae en el
  default `'binary'` cuando los labels son binarios. ✅ Efectivamente
  equivalente, pero para que sea **literalmente** el código descrito en
  el paper conviene pasarlo explícito.
- **Opciones**:
  - [x] **[REC]** Añadir `average='binary'` explícito en la llamada:
        `f1_score(y_true, y_pred, pos_label=1, labels=[0, 1], average='binary')`.
        Cambio cosmético pero elimina cualquier duda al lector del código.
  - [ ] Dejar como está.

### 6.4 Criterio de selección del grid search: F1 de instancia **medio**

- [ ] **Paper (§4.5)**: "maximizando el F1 de instancia medio" sobre
      los plots val.
- **Código** (`grid_search.py::run_grid_search`):
  `mean_f1 = np.mean(f1_matrix[combo_idx])` — media aritmética por plot.
  ✅ Consistente.

### 6.5 `min_tree_height` y `min_points_per_tree` fijos en grid search

- [ ] **Paper (Tabla 5)**: ambos fijos (2.0 m y 50 puntos).
- **Código**: ✅ toma los valores de `cfg.watershed.min_tree_height`
  y `cfg.watershed.min_points_per_tree` sin meterlos en el grid.

### 6.6 Tres dimensiones del grid coinciden

- [ ] **Paper (Tabla 5)**: `voxel_size ∈ {0.1, 0.2, 0.3}`,
      `gaussian_sigma ∈ {0.3, 0.5, 1.0}`, `min_crown_radius_m ∈ {0.5, 1.0, 1.5, 2.0}`,
      total 36 combinaciones.
- **Código** (`grid_search.py::GRID`): ✅ idéntico.

---

## 7. Detalles menores de reproducibilidad

### 7.1 Semilla global 42

- [ ] **Paper (§3.3)**: "random_state = 42".
- **Código**: ✅ `cfg.data.random_state = 42`, `cfg.rf.random_state = 42`,
  todas las fuentes de aleatoriedad la usan.
- **Excepción**: `predict_pointnet2.py` usa `np.random.choice` para el
  submuestreo de inferencia **sin fijar semilla local**. Esto significa
  que corridas repetidas de inferencia PN++ darán _resultados diferentes_
  por plot (aunque la media converge). El paper dice "todas las fuentes
  de aleatoriedad del pipeline ... usan la semilla fija 42 salvo
  indicación explícita en contrario".
- **Opciones**:
  - [x] **[REC]** Añadir en `predict_pointnet2.py` al principio de
        `predict_plot_pointnet2`: `rng = np.random.default_rng(42)` y usar
        `rng.choice(...)` en vez de `np.random.choice(...)`. Reproducibilidad
        completa sin cambio de comportamiento esperado.
  - [ ] Documentar explícitamente en §3.3 que la inferencia PointNet++
        no es determinística (el promedio converge pero cada corrida difiere).

### 7.2 Número de plots val tras train_val_split al 20%

- [ ] **Paper (§3.3)**: "80% / 20% a nivel de plot con semilla 42".
- **Código**: ✅ `sklearn.train_test_split(test_size=0.2, random_state=42)`.

### 7.3 Ordenamiento alfabético de plots en train_rf

- [ ] **Paper**: no lo exige.
- **Código** (`train_rf.py`): `sorted(train_paths, key=lambda p: p.stem)`.
  Es sólo para estabilidad de logs, no afecta resultados porque el
  subsample por plot tiene semilla derivada del stem.
- **Opciones**:
  - [ ] Ninguna. Detalle inofensivo.

### 7.4 Cache de features RF

- [ ] **Paper**: no lo menciona.
- **Código**: `output/features_cache/{plot}_features.npy`. Detalle de
  performance, no afecta resultados.
- **Opciones**: ninguna.

---

## 8. Puntos que SÍ coinciden (trazabilidad positiva)

Registro para no volver a revisar. Ninguna acción requerida.

- [x] Mapeo semántico `{1,2}→no-árbol, {4,5,6}→árbol, {0,3}→excluir`
      (paper §3.5 y §4.2 ↔ `dataset.py::get_binary_labels`).
- [x] Flujo A pasa `{1,2,4,5,6}` al watershed (label != -1) — paper §4.2 ↔
      `run_baseline.py`.
- [x] `hag_max = 50 m`, clamp a [0, 50] — paper §4.2 ↔ `config.yaml` y
      `ground_filter.py::normalize_height`.
- [x] HAG/hag_max normalizada a [0, 1] para PointNet++ — paper §4.2 ↔
      `pointnet2_dataset.py` y `predict_pointnet2.py`.
- [x] RF hiperparámetros `n_estimators=200`, `max_depth=None`,
      `class_weight=balanced`, `max_features=sqrt` (default sklearn) —
      paper Tabla 3 ↔ `config.yaml` y `train_rf.py`.
- [x] PointNet++: input dim 5 (XYZ + HAG + I), num_classes=2,
      Adam lr=1e-3, wd=1e-4, StepLR γ=0.7 cada 20 ep, 100 ep, bs=16,
      grad_clip=1.0, dropout=0.5 — paper Tabla 4 ↔ `config.yaml` y código.
- [x] Augmentation: jitter σ=0.02 en espacio normalizado, flip X p=0.5,
      scale [0.9, 1.1], sin rotación Z — paper §4.4 ↔ `pointnet2_dataset.py`.
- [x] Normalización per-sample: centroide + radio máximo — paper §4.4 ↔
      `pointnet2_dataset.py::__getitem__` y `predict_pointnet2.py`.
- [x] Densidad volumétrica con `cKDTree.query_ball_point` y r=0.5 m —
      paper §4.3 nota al pie ↔ `features_rf.py::compute_features_batch`.
- [x] Greedy matching instancia con τ=0.5 y tiebreak por índice —
      paper §5.2 ↔ `instance_metrics.py::match_instances`.
- [x] Filtro `HAG ≥ 0.5 m` sobre predicciones tree de los flujos B y C —
      paper §4.5 ↔ `predict_rf.py::filter_tree_points`.

---

## 9. Propuesta de orden de conciliación

Cuando toque aplicar cambios, yo iría en este orden (riesgo creciente,
esfuerzo creciente):

1. **Ediciones sólo-paper** (sin correr nada): §2.2, §3.1, §3.3, §3.4,
   §4.1, §4.2, §5.4.
2. **Ediciones sólo-código sin reentrenamiento**: §5.2, §6.3, §7.1,
   corrección docstring §3.1.
3. **Ediciones código que requieren recomputar cache RF y reentrenar**:
   §3.1 (omnivarianza). Solo vale la pena si el resultado cambia lo
   suficiente para alterar alguna conclusión. Mi corazonada: no lo hace,
   pero conviene medirlo.
4. **Ediciones que afectan la estructura del experimento**: §1.1
   (filtrado NIBIO2). Antes de aplicar, verificar que en
   `FORinstance_dataset/` no haya NIBIO2 actualmente — si no lo hay,
   la frase del paper es true _en esta corrida_ y sólo queda endurecer
   el código para que lo siga siendo.

---

_Generado: 2026-04-11._
_Convención de conciliación: decisiones en este archivo, cambios de código agrupados al final (flujo acordado)._
