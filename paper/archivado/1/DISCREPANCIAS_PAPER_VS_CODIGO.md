# Discrepancias Paper vs. Codigo

Documento de conciliacion entre `paper/main.tex` y la implementacion en `forest_its/`.

---

## 1. Asignacion de flujos B y C esta invertida

|             | Paper         | Codigo                            |
| ----------- | ------------- | --------------------------------- |
| **Flujo B** | PointNet++    | Random Forest (`methods/rf/`)     |
| **Flujo C** | Random Forest | PointNet++ (`methods/pointnet2/`) |

El paper define Flujo B = PointNet++ y Flujo C = Random Forest (lineas 186-189 de main.tex). El codigo y el README los manejan al reves: Metodo B = Random Forest, Metodo C = PointNet++. **Hay que decidir una convencion unica y alinear ambos.**

---

## 2. Detector ITD: paper dice "Raster Z-max", codigo usa Watershed 3D

El paper describe un detector **Raster Z-max 2D** con 4 pasos (Seccion 4.6): construir raster Z-max, suavizado gaussiano 2D, maximos locales 2D, emparejamiento con GT. El detector del paper produce **posiciones** de arboles (coordenadas del apex), no segmentaciones de instancia.

El codigo implementa **Watershed 3D volumetrico** (`segmentation/watershed3d.py`): voxelizacion 3D, suavizado gaussiano 3D, deteccion de semillas via CHM 2D + peak_local_max, watershed sobre volumen de densidad, y asignacion de puntos a instancias. El resultado del codigo son **IDs de instancia por punto**, no posiciones.

Esto es una discrepancia fundamental: son dos metodos distintos de segmentacion de instancias. Las metricas tambien cambian: el paper describe metricas de **deteccion** (emparejamiento apex-apex por distancia horizontal), mientras que el codigo calcula metricas de **segmentacion de instancias** (IoU 3D entre conjuntos de puntos, umbral IoU >= 0.5).

**Conciliar:** decidir cual de los dos metodos es el definitivo, y alinear paper y codigo.

---

## 3. Features del RF: paper dice 15 features basadas en bloque, codigo usa 28 features multi-escala

| Aspecto                    | Paper (Tabla 3, Sec. 4.5)                                                                                                                        | Codigo (`features_rf.py`)                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| **Numero de features**     | 15                                                                                                                                               | 28 (14 x 2 escalas)                                                                                                        |
| **Tipo**                   | Coordenadas (x,y,z), posicion vertical (z_rank_pct, z_above_mean, z_norm_block), distancia al centro (dist_center), estadisticas kNN k=16 y k=32 | 8 eigen-features de Weinmann + HAG, verticalidad, densidad, rugosidad, rango de altura, intensidad — a escalas k=20 y k=50 |
| **Escalas KNN**            | k=16 y k=32                                                                                                                                      | k=20 y k=50                                                                                                                |
| **Incluye intensidad**     | No (solo geometricas)                                                                                                                            | Si (feature #14)                                                                                                           |
| **Incluye HAG**            | No                                                                                                                                               | Si (feature #9)                                                                                                            |
| **Incluye eigen-features** | No                                                                                                                                               | Si (linearity, planarity, sphericity, etc.)                                                                                |
| **Usa bloques**            | Si (5m x 5m)                                                                                                                                     | No (calcula directamente sobre el plot completo)                                                                           |

Son dos conjuntos de features completamente diferentes. **Conciliar ambos.**

---

## 4. Paper describe segmentacion en bloques 5m x 5m para RF y PointNet++; codigo no usa bloques

El paper (Seccion 4.2, Tabla 2) describe una segmentacion en bloques de 5.0m x 5.0m con stride 2.5m y 4,096 puntos por bloque, aplicada a los flujos B y C. Indica numeros concretos: 2,902 bloques dev, 511 bloques val, 1,484 bloques test.

El codigo no implementa esta segmentacion en bloques:

- **RF** (`train_rf.py`): entrena directamente sobre las features por punto del plot completo (con submuestreo por clase). No hay bloques.
- **PointNet++** (`pointnet2_dataset.py`): submuestrea 8,192 puntos aleatoriamente del plot completo, no de bloques espaciales.
- La inferencia de PointNet++ (`predict_pn2.py`) usa submuestreo global repetido (multi-pass), no sliding window de bloques.

**Conciliar:** la estrategia de procesamiento por bloques del paper no existe en el codigo.

---

## 5. PointNet++ input: paper dice "solo XYZ (3 dim)", codigo usa XYZ + HAG + intensidad (5 dim)

| Aspecto                 | Paper (Tabla 4, linea 221)                                   | Codigo (`model_msg.py`)                                                    |
| ----------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------- |
| **Entrada**             | XYZ normalizado (3 dim.)                                     | XYZ + HAG_norm + intensidad (5 canales, `in_channel=5`)                    |
| **Justificacion paper** | "disponibilidad universal de XYZ en cualquier escaner LiDAR" | Wang et al. (2023) muestra mejora significativa al anadir HAG e intensidad |

El modelo esta construido con `in_channel=5` y recibe `features=(B, N, 2)` [HAG_norm, intensity]. Esto contradice directamente el paper.

---

## 6. Arquitectura PointNet++: radios y capas SA difieren

| Aspecto                | Paper (Tabla 4)                      | Codigo (`model_msg.py`)                                                                       |
| ---------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------- |
| **Capas SA (npoints)** | [1024, 256, 64] (3 capas)            | [1024, 256, 64, 16] (4 capas)                                                                 |
| **Radios**             | [0.2, 0.5, 1.0] m (1 radio por capa) | MSG con multiples radios por capa: SA1=[0.1,0.2], SA2=[0.2,0.4], SA3=[0.4,0.8], SA4=[0.8,1.6] |
| **Tipo**               | SSG (single scale) implicito         | MSG (multi-scale grouping) explicito                                                          |

El codigo tiene 4 capas SA con MSG (dos radios por capa), el paper describe 3 capas con un solo radio cada una.

---

## 7. Loss function de PointNet++: paper dice "CE + Dice", codigo usa NLLLoss

|          | Paper (Tabla 4)          | Codigo (`train_pn2.py`)            |
| -------- | ------------------------ | ---------------------------------- |
| **Loss** | CE + Dice (peso 0.5 c/u) | `nn.NLLLoss(weight=class_weights)` |

No hay componente Dice en el codigo. Solo se usa NLLLoss con pesos de clase.

---

## 8. Optimizer: paper dice AdamW, codigo usa Adam

|               | Paper (Tabla 4) | Codigo (`train_pn2.py`, linea 357) |
| ------------- | --------------- | ---------------------------------- |
| **Optimizer** | AdamW           | `optim.Adam`                       |

Adam y AdamW difieren en como aplican el weight decay (desacoplado vs. L2).

---

## 9. Puntos por muestra PointNet++: paper dice 4,096, config dice 8,192

|                        | Paper (Tabla 2) | Codigo (`config.yaml`, `pointnet2_dataset.py`) |
| ---------------------- | --------------- | ---------------------------------------------- |
| **Puntos por muestra** | 4,096           | 8,192 (`num_points: 8192`)                     |

---

## 10. RF max_depth: paper dice 20, config dice null

|               | Paper (Tabla 6) | Codigo (`config.yaml`)                    |
| ------------- | --------------- | ----------------------------------------- |
| **max_depth** | 20              | `null` (sin limite, crecimiento completo) |

Ademas, el docstring de `train_rf.py` dice "max_depth=None: crecimiento completo, el ensemble regulariza", contradiciendo el paper.

---

## 11. Paper describe RF entrenado por bloques con 1,024 pts/bloque; codigo entrena por plot completo

El paper (Seccion 4.5, parrafo "Submuestra de entrenamiento") indica que el RF entrena con 1,024 puntos por bloque, reduciendo de 11.8M a 2.97M puntos.

El codigo (`train_rf.py`) entrena directamente sobre puntos del plot completo con submuestreo balanceado por clase (`max_points_per_class: 500000`), sin concepto de bloques ni 1,024 puntos por bloque.

---

## 12. Metricas de evaluacion: el paper describe metricas ITD (deteccion), el codigo calcula metricas de instancia (IoU 3D)

| Aspecto                 | Paper (Seccion 5)                                                                                                           | Codigo (`instance_metrics.py`)                                                                            |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Tipo**                | Deteccion de arboles individuales (ITD): emparejamiento por distancia horizontal entre apex predicho y apex GT, umbral 3.0m | Segmentacion de instancias: emparejamiento greedy por IoU 3D entre conjuntos de puntos, umbral IoU >= 0.5 |
| **TP**                  | Maximo predicho dentro de 3.0m del apex GT                                                                                  | Instancia predicha con IoU >= 0.5 con instancia GT                                                        |
| **Salida del detector** | Posiciones (coordenadas)                                                                                                    | IDs de instancia por punto                                                                                |

Son protocolos de evaluacion fundamentalmente distintos. **Esta discrepancia esta ligada a la #2.**

---

## 13. Paper menciona "grid search independiente por flujo" para calibrar detector; no existe en codigo

El paper enfatiza (Seccion 4.6.4, Tabla 7) que los parametros del detector se calibran por grid search independiente por flujo sobre el conjunto dev. El codigo usa parametros fijos de `config.yaml` para el watershed, identicos para los tres metodos. No existe logica de grid search.

---

## 14. Paper no menciona 6 clases semanticas ni NIBIO2

El paper (Seccion 3, Tabla 1) describe 5 colecciones: NIBIO, CULS, TUWIEN, RMIT, SCION. El codigo (`dataset.py` docstring) y el README mencionan 6 colecciones: NIBIO, **NIBIO2**, CULS, SCION, RMIT, TUWIEN. El paper no menciona NIBIO2.

Ademas, el paper describe 6 clases semanticas (suelo, vegetacion baja, tronco, ramas vivas, ramas muertas, outpoints). El codigo define las clases como: 0=Unclassified, 1=Low-vegetation, 2=Terrain, 3=Out-points, 4=Stem, 5=Live-branches, 6=Woody-branches (7 categorias, con Unclassified como adicional no mencionada explicitamente en la lista del paper como clase separada).

---

## 15. El paper no menciona el uso de HAG como eje Z en el Watershed

En el codigo, tanto el baseline como los metodos B y C reemplazan la coordenada Z por el HAG antes de pasar los puntos al watershed:

```python
points_for_ws = np.column_stack([xyz[:, 0], xyz[:, 1], hag])
```

El paper no describe este detalle. Esto es metodologicamente relevante: usar HAG en vez de Z absoluto afecta directamente la deteccion de semillas y la segmentacion.

---

## Resumen de acciones

| #   | Prioridad   | Accion requerida                                                                          |
| --- | ----------- | ----------------------------------------------------------------------------------------- |
| 1   | Alta        | Decidir convencion Flujo B/C y alinear paper <-> codigo                                   |
| 2   | **Critica** | Decidir detector definitivo: Raster Z-max (paper) vs. Watershed 3D (codigo)               |
| 3   | Alta        | Alinear features del RF: 15 por bloque (paper) vs. 28 multi-escala (codigo)               |
| 4   | Alta        | Alinear estrategia de bloques vs. plot completo                                           |
| 5   | Alta        | Alinear input de PointNet++: 3 dim (paper) vs. 5 dim (codigo)                             |
| 6   | Media       | Alinear arquitectura SA layers: 3 SSG (paper) vs. 4 MSG (codigo)                          |
| 7   | Media       | Alinear loss: CE+Dice (paper) vs. NLLLoss (codigo)                                        |
| 8   | Baja        | Alinear optimizer: AdamW (paper) vs. Adam (codigo)                                        |
| 9   | Media       | Alinear num_points: 4096 (paper) vs. 8192 (codigo)                                        |
| 10  | Media       | Alinear max_depth RF: 20 (paper) vs. null (codigo)                                        |
| 11  | Alta        | Alinear entrenamiento RF: bloques (paper) vs. plot completo (codigo)                      |
| 12  | **Critica** | Alinear protocolo de evaluacion: ITD por distancia (paper) vs. IoU 3D instancias (codigo) |
| 13  | Alta        | Implementar grid search o remover del paper                                               |
| 14  | Baja        | Mencionar NIBIO2 si aplica; verificar conteo de clases                                    |
| 15  | Media       | Documentar uso de HAG como Z en watershed en el paper                                     |
