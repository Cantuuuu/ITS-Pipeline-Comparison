# Discrepancias paper ↔ código

Revisión sistemática del manuscrito `paper/main.tex` contra la
implementación en `forest_its/`. Cada entrada documenta lo que afirma el
paper, lo que hace realmente el código y la severidad de la divergencia.
**No se aplican cambios todavía** — este archivo es solo el registro de
decisiones por tomar (ver flujo de conciliación en memoria de usuario).

Severidades:
- **alta**: el paper afirma algo que el código no hace; afecta
  reproducibilidad o resultados numéricos.
- **media**: descripción imprecisa o ambigua; el código hace algo
  razonable pero no exactamente lo descrito.
- **baja**: cosmética / docstrings / comentarios desactualizados, sin
  impacto en resultados.

---

## D1 — Combinación de la semilla por plot del Random Forest [alta]

**Paper** §3.3, líneas 174–175 (`paper/main.tex:175`):
> "...la semilla efectiva se deriva como la **combinación bit a bit**
> de la semilla global con el CRC32 del nombre del plot..."

**Paper** §4.4, línea 244 (`paper/main.tex:244`):
> "...la semilla efectiva de cada plot se deriva combinando
> `random_state=42` con el CRC32 del nombre del archivo..."

**Código** `forest_its/methods/rf/train_rf.py:145-147`:
```python
plot_seed = (
    cfg.rf.random_state + zlib.crc32(las_path.stem.encode("utf-8"))
) & 0x7FFFFFFF
```
Es **suma** + máscara, no XOR (ni ninguna combinación bit a bit).

**Docstring** `forest_its/methods/rf/train_rf.py:17-19` añade una tercera
versión:
> "La semilla por plot `crc32(stem) XOR random_state`..."

Tres descripciones diferentes (paper: bitwise; docstring: XOR;
implementación: suma).

**Decisión pendiente:** elegir una sola convención. Como `42` es chico
respecto al CRC32, suma vs XOR producen seeds prácticamente equivalentes
en distribución, pero los seeds concretos son distintos → cualquier
cambio invalida los resultados anteriores. Recomendado: dejar el código
como está (`+`) y corregir paper + docstring para reflejarlo.

---

## D2 — Tie-breaking del greedy matching de instancias [alta]

**Paper** §5.2, líneas 396–401 (`paper/main.tex:400`):
> "Los pares restantes se ordenan por IoU descendente. **En caso de
> empate exacto se desempata por el orden natural de los identificadores
> (primero por menor pred_id, después por menor gt_id) para garantizar
> determinismo.**"

**Código** `forest_its/evaluation/instance_metrics.py:100`:
```python
flat_indices = np.argsort(-iou_matrix.ravel())
```
`np.argsort` con `kind='quicksort'` (default) **no es estable** y no
implementa ningún tie-break secundario por `(pred_id, gt_id)`. El
resultado es determinístico para una entrada fija pero no respeta el
orden prometido en el paper.

**Decisión pendiente:** o bien cambiar el código a `np.lexsort` /
`argsort(kind='stable')` con keys `(pred_id, gt_id, -IoU)`, o bien
relajar la frase del paper a "el orden de empates queda determinado por
`np.argsort`, que es determinístico para una matriz de IoU dada".

---

## D3 — Feature #11 (densidad local) está limitada por k vecinos [alta]

**Paper** Tabla 4, fila 11 (`paper/main.tex:232`):
> "Conteo de vecinos local — `|{p_j : ||p_j − p_i|| ≤ r}|` (cardinalidad)"

**Paper** §4.4, línea 241 (`paper/main.tex:241`):
> "La feature #11 se computa como el número entero de vecinos del punto
> central dentro de una esfera de radio fijo r = density_radius = 0,5 m,
> idéntico para todos los plots y ambas escalas."

Esto promete cardinalidad real de la bola de radio `r`.

**Código** `forest_its/preprocessing/features_rf.py:184-202` y
`compute_local_features` líneas 149-153:
```python
nbr_idx = indices_chunk[local_i, 1:k + 1]   # solo k vecinos
...
neighbors_xyz = xyz[nbr_idx]
...
dists = np.linalg.norm(neighbors_xyz - point_xyz, axis=1)
density = float(np.sum(dists <= density_radius))
```
La densidad **solo cuenta vecinos dentro del top-k**, no todos los puntos
de la nube en el radio. Por eso:
- Está acotada por arriba en `k_small=20` para la primera escala y en
  `k_large=50` para la segunda.
- En zonas de alta densidad satura.
- **El valor difiere entre las dos escalas** (k20 vs k50), aunque el
  paper dice explícitamente que es "idéntico para todos los plots y
  ambas escalas".

**Decisión pendiente:** dos opciones limpias:
1. Cambiar el código a una `query_ball_point` real del KDTree para tener
   cardinalidad sin tope, computada una sola vez por punto.
2. Reescribir la fila #11 del paper como "conteo de vecinos del top-k
   dentro de la esfera de radio r" y reconocer que el valor depende de
   la escala (en cuyo caso ya no es la misma feature en k=20 y k=50 y
   habría que renombrarlas).

Esta es probablemente la discrepancia más sustantiva: cambia el contenido
informativo de la feature.

---

## D4 — Tipo de filtro de suavizado del DTM [media]

**Paper** §4.2, línea 206 (`paper/main.tex:206`):
> "...rasterizado a una resolución horizontal de 0,5 m, e interpolado
> con un suavizado de **ventana 5 × 5** para reducir artefactos
> locales."

**Código** `forest_its/preprocessing/ground_filter.py:129`:
```python
dtm_grid = uniform_filter(dtm_grid.astype(np.float64), size=smooth_window)
```
Es un **filtro media (uniform)** 5×5, no un gaussiano ni un kernel
explícito. El paper no especifica el tipo, así que técnicamente no
contradice — pero "suavizado de ventana 5×5" se suele leer como
gaussiano.

**Decisión pendiente:** añadir al paper "filtro media (uniform) de
ventana 5×5" para que sea exacto.

---

## D5 — Construcción del DTM: fallback no documentado [media]

**Paper** §4.2, línea 206:
> "El DTM utilizado para calcular la HAG se construye **por plot a
> partir de los puntos clasificados como suelo**..."

**Código** `forest_its/preprocessing/ground_filter.py:54-63`:
```python
if classification is not None:
    terrain_mask = classification == 2
    if terrain_mask.sum() > 100:
        xyz_ground = xyz[terrain_mask]
        use_terrain_class = True
    else:
        xyz_ground = xyz   # ← FALLBACK no documentado
```
Si hay menos de 100 puntos de clase 2, el código cae a "todos los
puntos" y usa percentil 5 por celda como aproximación al suelo. No es
mencionado en el paper.

Adicionalmente, en el caso "happy path" el código toma la **mediana** de
los puntos suelo por celda (`ground_filter.py:100`), no el mínimo ni el
percentil 5. El paper tampoco menciona este detalle.

**Decisión pendiente:** decidir si el paper debe documentar (a) el uso
de la mediana cuando hay clase 2 y (b) el fallback a percentil 5 sin
clase 2. Como FOR-instance siempre tiene clase 2 anotada, el fallback
nunca se activa en este experimento; podría documentarse en el apéndice
o eliminarse del código.

---

## D6 — `seg F1` reportado vs definido [media]

**Paper** §5.1, líneas 379–383 (`paper/main.tex:383`):
> "seg F1 — F1 binario de la clase positiva (árbol)... En la
> implementación esto corresponde a `sklearn.metrics.f1_score` con
> `average='binary'` y `pos_label=1`."

**Código** `forest_its/evaluation/semantic_metrics.py:57-58`:
```python
f1_w = float(f1_score(y_true, y_pred, average="weighted", labels=[0, 1]))
f1_tree = float(f1_score(y_true, y_pred, pos_label=1, labels=[0, 1]))
```
Calcula ambos. La métrica `f1_tree` (binaria pos_label=1) sí coincide
con la definición del paper.

**Pero** los pipelines (`run_rf_pipeline.py:325`, `run_pn2_pipeline.py:345`)
imprimen como "F1 weight" la columna `sem_f1_weighted`, no
`sem_f1_tree`. Si la tabla final del paper se llena leyendo
`sem_f1_weighted` del CSV, el número reportado **no será** el definido
en el paper.

**Decisión pendiente:** asegurar que la generación de la Tabla 7
(`tab:results-agg`) lee `sem_f1_tree`, no `sem_f1_weighted`. El logging
puede dejarse como está, pero conviene renombrar la línea de log a
"F1 (weighted)" para evitar confusión.

---

## D7 — Coverage = recall en la implementación [baja]

**Paper** §5.2, línea 406:
> "...coverage (cobertura), este último definido como la fracción de
> árboles GT que reciben al menos un emparejamiento sobre el umbral,
> **equivalente al recall sin penalización por sobre-segmentación**."

**Código** `forest_its/evaluation/instance_metrics.py:118`:
```python
coverage = tp / n_gt
```
con `tp = len(matched_gt)` y `n_gt = |GT|`. Eso es **exactamente** el
recall (`tp / (tp + fn) = tp / n_gt` cuando todos los GT no emparejados
cuentan como FN). Por construcción del greedy matching, coverage y
recall son matemáticamente idénticos en este código.

La frase "equivalente al recall sin penalización por sobre-segmentación"
sugiere que podrían diferir; en el código no difieren. Se puede:
1. Reformular el paper a "coverage es el mismo recall reportado en la
   columna anterior"; o
2. Definir coverage como "GT con ≥1 pred sobre umbral, sin imponer
   matching único" (lo que sí lo haría diferir del recall) y modificar
   el código en consecuencia.

**Decisión pendiente:** elegir interpretación (preferible (1)).

---

## D8 — Hardware en docstrings de PointNet++ [baja]

**Paper** Tabla 3, línea 295 (`paper/main.tex:295`):
> "Hardware Apple M4 Pro (memoria unificada)"

**Código** `forest_its/methods/pointnet2/train_pn2.py:15`:
```
- Verifica que cabe en VRAM (RTX 4050, 6GB)
```
y `forest_its/methods/pointnet2/predict_pn2.py:27`:
```
≈ 4.5 min para 5 plots val en RTX 4050.
```
Comentarios obsoletos referencian una RTX 4050, hardware distinto al
declarado en el paper. El config (`batch_size: 16`,
`mixed_precision: bf16`) está alineado con el M4 Pro.

**Decisión pendiente:** actualizar comentarios del código a M4 Pro
(o eliminarlos) para evitar dudas al revisor.

---

## D9 — Paper menciona "n_passes ≈ 99% cobertura" vs docstring 86% [baja]

**Paper** §4.4, línea 313:
> "Con este esquema la cobertura efectiva supera el 99% de los puntos
> del plot en todos los casos observados."

**Código** `forest_its/methods/pointnet2/predict_pn2.py:23-25`:
```
Cobertura esperada con n_passes = N / num_points * 3:
  Para N=2M, num_points=8192: n_passes=732 → E[cobertura/punto] ≈ 86%
```
Para `n = 3·N / N_s` la prob teórica de no ser muestreado tiende a
`exp(-3) ≈ 5%`, lo que da cobertura ≈ 95% antes de la interpolación KNN
final. El número 86% del docstring está mal calculado; no contradice al
paper sino al cálculo teórico.

**Decisión pendiente:** corregir el docstring con la cifra correcta
(~95% antes de KNN, >99% después). No afecta resultados.

---

## D10 — Hardware/tiempos del apéndice todavía PENDIENTES [baja]

**Paper** Tabla 8 (`paper/main.tex:730-746`):
- `Hardware PointNet++  & [COMPLETAR: GPU, memoria, tiempo total]`
- `Código & [PENDIENTE: URL repositorio]`

Pendientes que el paper marca como `[COMPLETAR]` y que tienen valores
inferibles del código/config:
- `cfg.pointnet2.epochs = 100`, `batch_size = 16`, M4 Pro → llenar la
  fila de hardware.
- URL del repo: cuando exista.

**Decisión pendiente:** completar a mano cuando se tengan los tiempos
medidos en M4 Pro.

---

## D11 — Densidades de puntos por sitio sin completar [baja]

**Paper** §3.1, línea 141:
> "...densidades que van de [COMPLETAR: rango por sitio] pts/m²..."

y §6.1 limitación 4 (`paper/main.tex:598`):
> "...densidades de [COMPLETAR] pts/m²."

El docstring de `forest_its/segmentation/watershed3d.py:9-10` cita
"~18,236 pts/m² (Wielgosz et al. 2024, SegmentAnyTree)" — habría que
verificar y reportar el rango real por colección antes de la submisión.

**Decisión pendiente:** completar o eliminar las menciones marcadas como
`[COMPLETAR]`.

---

## D12 — Naming inconsistente del flujo C en CLI [baja]

**Paper** se refiere consistentemente a "Flujo C — PointNet++ MSG" y a
los comandos que usan el módulo `pointnet2`.

**Código** mezcla dos nombres para el mismo flujo:
- Carpeta: `forest_its/methods/pointnet2/`
- Comandos: `python -m forest_its.methods.pointnet2.run_pn2_pipeline`
- Argumento de `grid_search.py`: `--methods ... pn2` (no `pointnet2`)
- CSV de predicciones: `output/predictions/pn2/`
- CSV de métricas: `pn2_metrics_{split}.csv`
- `CLAUDE.md` ejemplifica: `--methods baseline rf pointnet2` (que
  buscaría `pointnet2_metrics_*.csv`, archivo que **no existe**).

No es contradictorio con el paper (el paper no menciona `pn2` ni
`pointnet2` como nombre interno), pero sí entre código y `CLAUDE.md`.
Vale la pena unificar antes de publicar el repo.

**Decisión pendiente:** unificar en un solo nombre (`pn2` o `pointnet2`)
en todos los CSVs, CLIs y `CLAUDE.md`.

---

## Resumen de severidades

| ID  | Tema                                          | Severidad |
|-----|-----------------------------------------------|-----------|
| D1  | Operador combinación de seed RF               | alta      |
| D2  | Tie-breaking greedy matching                  | alta      |
| D3  | Feature #11 acotada por k                     | alta      |
| D4  | Tipo de filtro DTM (uniform vs gaussiano)     | media     |
| D5  | Fallback DTM sin clase 2 + uso de mediana     | media     |
| D6  | `seg F1` reportado: tree vs weighted          | media     |
| D7  | Coverage = recall en la implementación        | baja      |
| D8  | Comentarios PN++ con RTX 4050                 | baja      |
| D9  | Cobertura 86% en docstring vs ~95% real       | baja      |
| D10 | Apéndice tabla 8 todavía PENDIENTE            | baja      |
| D11 | Rango de densidades por sitio PENDIENTE       | baja      |
| D12 | Naming `pn2` vs `pointnet2`                    | baja      |

**Próximo paso sugerido:** discutir D1, D2, D3 una por una; cada una
implica una decisión de "fix code" vs "fix paper" antes de tocar
archivos.
