# Comparacion de Metodos de Segmentacion de Arboles Individuales con FOR-instance

## Pregunta de investigacion

> El preprocesamiento semantico (clasico vs. deep learning) mejora la calidad de la segmentacion de instancias cuando el segmentador de instancias es identico en los tres metodos?

## Pipelines

```
Metodo A - Baseline (sin preprocesamiento semantico)
  Nube normalizada (excl. clases 0,3) --> Watershed 3D --> Instancias

Metodo B - Random Forest + Watershed
  Nube normalizada --> 27 features --> RF (arbol/no-arbol)
  --> Filtrar "arbol" --> Watershed 3D --> Instancias

Metodo C - PointNet++ MSG + Watershed
  Nube normalizada --> PointNet++ MSG (arbol/no-arbol)
  --> Filtrar "arbol" --> Watershed 3D --> Instancias

                  +-------------------+
                  | Watershed 3D      |  <-- COMPARTIDO por A, B, C
                  | (identico modulo) |
                  +-------------------+
```

La unica variable independiente es la calidad del preprocesamiento semantico.

## Instalacion

```bash
conda env create -f environment.yml
conda activate forest_its
```

## Verificacion del dataset

```bash
python scripts/explore_dataset.py
```

Esto lee `data_split_metadata.csv`, carga un plot por carpeta institucional
(NIBIO, NIBIO2, CULS, SCION, RMIT, TUWIEN) e imprime estadisticas:
puntos totales, distribucion de clases, arboles unicos, densidad, etc.

## Orden de ejecucion

El watershed 3D comparte hiperparametros entre flujos, pero los puntos de
entrada difieren (todos los puntos validos vs. solo los clasificados como
arbol). Para una comparacion justa, cada flujo calibra sus parametros de
watershed mediante grid search sobre el split val. El flujo de trabajo es:

```
  1. Entrenar semanticos    (RF, PointNet++)
  2. Correr stage semantic  (genera semantic_pred sobre val)
  3. Grid search            (calibra watershed params por flujo sobre val)
  4. Stage instance val     (metricas val con best params)
  5. Stage instance test    (metricas test finales con best params)
```

### 1. Explorar dataset
```bash
python scripts/explore_dataset.py
```

### 2. Entrenar modelos semanticos
```bash
# Random Forest (Flujo B)
python -m forest_its.methods.rf.train_rf

# PointNet++ MSG (Flujo C)
python -m forest_its.methods.pointnet2.train_pointnet2
```

### 3. Stage semantic — generar predicciones semanticas val
```bash
# Flujo B: RF
python -m forest_its.methods.rf.run_rf_pipeline --stage semantic --split val

# Flujo C: PointNet++
python -m forest_its.methods.pointnet2.run_pointnet2_pipeline --stage semantic --split val
```

(El Flujo A baseline no tiene preprocesamiento semantico — salta este paso.)

### 4. Grid search de watershed params por flujo
```bash
python -m forest_its.evaluation.grid_search --methods baseline rf pointnet2
```

Produce `output/results/grid_search_best_params.csv`, consultado por los
pipelines en la stage instance. Sin este paso los pipelines fallan con
`MissingBestParamsError`.

### 5. Stage instance — metricas val (con best params)
```bash
# Flujo A: Baseline
python -m forest_its.methods.baseline.run_baseline --split val

# Flujo B: RF
python -m forest_its.methods.rf.run_rf_pipeline --stage instance --split val

# Flujo C: PointNet++
python -m forest_its.methods.pointnet2.run_pointnet2_pipeline --stage instance --split val
```

### 6. Stage instance — metricas test finales
```bash
python -m forest_its.methods.baseline.run_baseline --split test
python -m forest_its.methods.rf.run_rf_pipeline --stage instance --split test
python -m forest_its.methods.pointnet2.run_pointnet2_pipeline --stage instance --split test
```

### 7. Visualizacion
```bash
python scripts/visualize_results.py
```

## Features del Random Forest (28 = 14 x 2 escalas)

| #  | Feature              | Formula                        | Justificacion                         |
|----|----------------------|--------------------------------|---------------------------------------|
| 1  | Linearidad           | (l1-l2)/l1                     | Weinmann et al. 2017 - troncos        |
| 2  | Planaridad           | (l2-l3)/l1                     | Weinmann et al. 2017 - suelo          |
| 3  | Esfericidad          | l3/l1                          | Weinmann et al. 2017 - copa           |
| 4  | Omnivarianza         | (l1*l2*l3)^(1/3)               | Weinmann et al. 2017                  |
| 5  | Anisotropia          | (l1-l3)/l1                     | Weinmann et al. 2017                  |
| 6  | Eigenentropia        | -sum(li*ln(li))                | Weinmann et al. 2017                  |
| 7  | Suma eigenvalores    | l1+l2+l3 (crudos)              | Weinmann et al. 2017                  |
| 8  | Cambio curvatura     | l3/(l1+l2+l3)                  | Weinmann et al. 2017                  |
| 9  | HAG                  | Z - DTM(x,y)                   | Bremer 2023, Li 2023 - peso=0.21      |
| 10 | Verticalidad         | 1 - abs(dot(n,[0,0,1]))        | Complementa HAG                       |
| 11 | Densidad local       | k / vol_esfera                 | Contexto local                        |
| 12 | Rugosidad            | std(Z vecinos)                 | Varianza altura local                 |
| 13 | Rango de altura      | max(Z)-min(Z) vecinos          | Span vertical                         |
| 14 | Intensidad norm.     | intensity / max                | Fang 2022, Li 2023                    |

Escalas: k=20 (detalle fino) y k=50 (contexto grueso), siguiendo Weinmann 2015.
Eigenvalores normalizados para features 1-6 y 8: li = li/(l1+l2+l3).
Feature 7 usa eigenvalores crudos.

## Referencias

- Puliti et al. (2023). FOR-instance: a UAV laser scanning benchmark dataset
  for semantic and instance segmentation of individual trees.
- Weinmann et al. (2017). Geometric Features and their Relevance for 3D Point
  Cloud Classification. ISPRS Annals.
- Qi et al. (2017). PointNet++: Deep Hierarchical Feature Learning on Point
  Sets in a Metric Space. NeurIPS.
- Wielgosz et al. (2024). SegmentAnyTree: A sensor and platform agnostic deep
  learning model for tree segmentation using laser scanning data.
- Henrich et al. (2024). ForAINet: Deep learning for forest point cloud
  semantic and instance segmentation.
- Bremer et al. (2023). Feature importance analysis for LiDAR forest classification.
- Li et al. (2023). Point cloud classification in forestry using RF features.
