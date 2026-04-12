# Discrepancias Paper vs Codigo (v5 — 2026-04-12)

Revision sistematica del estado actual de `main.tex` contra el codigo en `forest_its/`.
Solo se listan discrepancias con impacto academico (reproducibilidad, rigor metodologico, revision por pares).

---

## 1. Arquitectura PointNet++ (SA/FP) no documentada en el paper

- [x] **Conciliado**

**Paper** (L273-274, Table `tab:pointnet`): Dice "4 modulos Set Abstraction, 4 Feature Propagation, implementacion basada en yanx27" pero NO lista radios, npoints, nsamples ni tamanhos de MLP por capa.

**Codigo** (`model_msg.py:40-50`):
```
SA1: npoint=1024, radii=[0.1,0.2], nsamples=[16,32], MLPs=[[16,16,32],[32,32,64]]  -> 96ch
SA2: npoint=256,  radii=[0.2,0.4], nsamples=[16,32], MLPs=[[64,64,128],[64,96,128]] -> 256ch
SA3: npoint=64,   radii=[0.4,0.8], nsamples=[16,32], MLPs=[[128,128,256],[128,128,256]] -> 512ch
SA4: npoint=16,   radii=[0.8,1.6], nsamples=[16,32], MLPs=[[256,256,512],[256,384,512]] -> 1024ch
FP4: in=1536, mlp=[512,512]
FP3: in=768,  mlp=[512,256]
FP2: in=352,  mlp=[256,128]
FP1: in=133,  mlp=[128,128]
```

**Impacto**: Un revisor no puede reproducir la red sin leer el codigo. Los radios en espacio normalizado son especificos de esta implementacion y difieren del repositorio original de yanx27 (que usa SSG con radios distintos).

**Opciones**:
- **(A)** Anadir tabla completa de arquitectura SA+FP al paper (Apendice o seccion 4.4).
- **(B)** Anadir los radios y npoints como nota al pie de la tabla `tab:pointnet` existente.
- **(C)** Dejar como esta y referenciar el codigo del repositorio publico.

**Recomendacion**: **(A)** — tabla en apendice. Los radios normalizados son decisiones no triviales y la referencia a yanx27 es insuficiente porque el repositorio usa SSG, no MSG con estos radios especificos. Una tabla de ~10 filas no ocupa espacio excesivo en LNCS y cierra la reproducibilidad.

---

## 2. Divisor de covarianza (k vs k-1) no documentado

- [x] **Conciliado**

**Paper**: No menciona que divisor se usa en la matriz de covarianza para los eigenfeatures.

**Codigo** (`features_rf.py:91`): `cov = (centered.T @ centered) / k` — divisor k (no k-1).

**Impacto**: La convencion k (vs la covarianza muestral con k-1) es una decision explicita alineada con Weinmann (2014). Un revisor de estadistica podria cuestionar por que no se usa k-1. Documentarlo previene la pregunta.

**Opciones**:
- **(A)** Anadir una frase al parrafo de features: "La covarianza local se estima con divisor $k$, consistente con la convencion de Weinmann et al. (2014, 2015) para features de point cloud."
- **(B)** Dejar sin mención.

**Recomendacion**: **(A)** — una frase. Costo cero en espacio, previene revision innecesaria.

---

## 3. Formula de pesos de NLLLoss no especificada

- [x] **Conciliado**

**Paper** (L292, L306): "NLLLoss ponderada por la frecuencia inversa de cada clase".

**Codigo** (`train_pointnet2.py`): `weights = total / (2 * counts + 1e-6)` — el factor 2 produce pesos proporcionales a `N / (2 * N_c)`, no estrictamente `1 / N_c`.

**Impacto**: "Frecuencia inversa" es ambiguo: `1/freq`, `N/N_c`, `N/(C*N_c)` son todas variantes validas. La formula especifica afecta el balance del gradiente y por tanto los resultados. Un lector que implemente la formula naive `1/N_c` obtendra pesos diferentes.

**Opciones**:
- **(A)** Documentar la formula exacta en el paper: $w_c = N / (2 N_c + \epsilon)$ con $\epsilon = 10^{-6}$.
- **(B)** Describir como "inversamente proporcional al conteo de cada clase, normalizado por el numero total de clases" (que es lo que hace el factor 2).

**Recomendacion**: **(A)** — la formula es una linea de LaTeX. Elimina ambiguedad.

---

## 4. `max_features` no explicito en el codigo

- [x] **Conciliado**

**Paper** (Table `tab:rf-hp`, L262): `max_features = sqrt`.

**Codigo** (`train_rf.py:183-190`): `RandomForestClassifier(n_estimators=..., max_depth=..., class_weight=..., ...)` — NO pasa `max_features`. Depende del default de sklearn (que es `"sqrt"` desde sklearn 1.0).

**Impacto**: El behavior actual es correcto (sklearn >= 1.4 garantizado por environment.yml). Pero si alguien ejecuta con sklearn < 1.0 (donde el default era `"auto"` / `n_features`), el modelo cambiaria. Riesgo bajo pero facil de cerrar.

**Opciones**:
- **(A)** Anadir `max_features="sqrt"` al constructor de RF en el codigo.
- **(B)** Anadir `max_features` al config.yaml y pasarlo al constructor.

**Recomendacion**: **(A)** — una linea de codigo. Hace explicito lo que el paper promete.

---

## 5. Agregacion de metricas: consola usa promedio ponderado, paper dice sin ponderar

- [x] **Conciliado**

**Paper** (L422): "Las metricas se computan por plot y luego se promedian sobre los plots del split."

**Codigo** (`run_evaluation.py:83-116`): La salida por consola ("Global averages") usa **promedio ponderado por n_gt_trees** para precision, recall, F1 y coverage. El generador LaTeX (`_write_latex_table`, L147-153) usa `.mean()` **sin ponderar**.

**Impacto**: Si los valores reportados en el paper se copian de la consola (y no del LaTeX generado), seran ponderados, no promedios simples como dice el paper. Con el desbalance de arboles entre sitios (NIBIO=161 vs CULS=20), la diferencia puede ser no trivial.

**Opciones**:
- **(A)** Unificar codigo: que la consola tambien imprima promedios sin ponderar (consistente con paper y con LaTeX).
- **(B)** Cambiar el paper a promedio ponderado y ajustar consola + LaTeX.
- **(C)** Reportar ambos (ponderado y no ponderado) en paper y codigo.

**Recomendacion**: **(A)** — la practica estandar en FOR-instance (ForAINet, SegmentAnyTree) es promedio sin ponderar por plot. Alinear la consola con el LaTeX generado y con el paper.

---

## 6. Entrenamiento PointNet++ sin semillas (reproducibilidad rota)

- [x] **Conciliado**

**Paper** (L175, Apendice): "random_state = 42 (global)" para todas las fuentes de aleatoriedad.

**Codigo** (`train_pointnet2.py`): NO llama `np.random.seed()`, `torch.manual_seed()`, ni `random.seed()` al inicio del entrenamiento. Consecuencias:
- Inicializacion de pesos: no determinista.
- Augmentation en `ForInstanceDataset.__getitem__`: usa `np.random.normal()`, `np.random.random()`, `np.random.uniform()` con el RNG global de numpy.
- DataLoader shuffle: no determinista.
- Resultado: dos corridas de entrenamiento producen modelos diferentes.

**Impacto**: El paper promete reproducibilidad completa con seed 42. En la practica, el entrenamiento de PointNet++ no es reproducible. Un revisor que intente replicar los resultados obtendra metricas diferentes.

**Opciones**:
- **(A)** Anadir al inicio de `main()` en `train_pointnet2.py`:
  ```python
  seed = cfg.data.random_state  # 42
  np.random.seed(seed)
  torch.manual_seed(seed)
  random.seed(seed)
  ```
  Y migrar augmentation de `np.random.*` global a `np.random.default_rng(worker_seed)` con `worker_init_fn` en el DataLoader.
- **(B)** Documentar en el paper que el entrenamiento de PointNet++ no es bit-exact reproducible (solo la particion train/val y la inferencia lo son), y que la reproducibilidad reportada se refiere a la semilla de splitting y evaluacion.

**Recomendacion**: **(A)** — fijar semillas es el estandar minimo. El DataLoader multi-worker con numpy global RNG es un bug conocido; la solucion canonica es `worker_init_fn + Generator`. Ademas, anadir una nota al paper indicando que la reproducibilidad bit-exact del entrenamiento depende de la plataforma (MPS vs CUDA vs CPU) pero que los seeds se fijan para maxima consistencia.

---

## 7. Augmentation usa RNG global de numpy (no thread-safe)

- [x] **Conciliado**

**Relacionada con #6** pero es un problema independiente.

**Codigo** (`pointnet2_dataset.py:141-147`): `np.random.normal(...)`, `np.random.random()`, `np.random.uniform(...)` usan el RNG global de numpy.

**Impacto**: Con `num_workers > 0` en el DataLoader, cada worker hereda una copia del estado del RNG global. Sin `worker_init_fn`, todos los workers generan las mismas secuencias de augmentation, lo que reduce la variabilidad efectiva de la augmentation. Con `num_workers=6`, hay potencialmente 6 workers produciendo jitter identico en cada batch.

**Opciones**:
- **(A)** Migrar a `np.random.default_rng()` local por sample, inicializado en `worker_init_fn` con seed derivado del worker_id y la epoca.
- **(B)** Migrar a augmentation con PyTorch tensors (`torch.randn`, `torch.rand`), que usa el RNG de PyTorch per-worker correctamente.

**Recomendacion**: **(A)** — `np.random.default_rng(base_seed + worker_id * epoch)` en `worker_init_fn` y uso de un `rng` local en `__getitem__`. Es la solucion PyTorch canonica.

---

## 8. Wang et al. (2023) citado en codigo pero no en paper

- [x] **Conciliado**

**Paper** (L276-277): La justificacion de la entrada 5D (XYZ+HAG+I) se basa en "consistencia con las senales del Flujo B".

**Codigo** (`model_msg.py:10-12`): Cita "Wang et al. (2023) Remote Sens., que demuestra mejora significativa al anadir altura normalizada e intensidad sobre XYZ puro en clasificacion forestal con PointNet++."

**Impacto**: La referencia Wang et al. (2023) proporciona evidencia empirica adicional para la eleccion 5D, independiente del argumento de consistencia. Incluirla en el paper fortaleceria la justificacion ante un revisor que pregunte por que no usar solo XYZ.

**Opciones**:
- **(A)** Anadir Wang et al. (2023) al paper como referencia complementaria en el parrafo de justificacion 5D.
- **(B)** Dejar solo el argumento de consistencia (el paper ya tiene una justificacion valida).

**Recomendacion**: **(A)** — citar evidencia experimental nunca debilita un argumento. Una frase: "Esta eleccion es ademas consistente con Wang et al. (2023), que reporta mejoras significativas al incorporar HAG e intensidad en PointNet++ para clasificacion forestal."

---

## Resumen

| # | Discrepancia | Severidad | Esfuerzo |
|---|-------------|-----------|----------|
| 1 | Arquitectura SA/FP no en paper | Alta (reproducibilidad) | Media (tabla apendice) |
| 2 | Divisor covarianza k no documentado | Baja (correccion) | Baja (1 frase) |
| 3 | Formula pesos NLLLoss no especificada | Media (reproducibilidad) | Baja (1 ecuacion) |
| 4 | max_features no explicito en codigo | Baja (robustez) | Baja (1 linea codigo) |
| 5 | Agregacion metricas inconsistente | Media (validez resultados) | Baja (alinear consola) |
| 6 | PN++ training sin semillas | Alta (reproducibilidad) | Media (seeds + worker_init) |
| 7 | Augmentation con RNG global | Media (correctitud training) | Media (migrar a local RNG) |
| 8 | Wang et al. 2023 no citado en paper | Baja (completitud) | Baja (1 referencia) |
