"""
Watershed 3D para segmentación de árboles individuales.

Este módulo es COMPARTIDO por los tres métodos del paper. La elección de
Watershed 3D sobre Watershed 2D (basado en CHM) se justifica porque:

1. El Watershed 2D colapsa la información vertical y falla en bosques densos
   donde las copas se superponen en proyección pero tienen troncos distintos.
2. FOR-instance contiene nubes de alta densidad ULS (ver §3.1 del paper
   para el rango medido sobre las colecciones empleadas), lo que hace
   viable el enfoque volumétrico sobre vóxeles pequeños.
3. Al evaluar si el preprocesamiento semántico aporta valor, ese efecto es
   más visible en 3D que en 2D donde el baseline ya es competitivo.

La única diferencia entre métodos es qué puntos se le pasan:
  - Método A (Baseline): todos los puntos con label != -1
  - Método B (RF):       solo puntos con pred_binary == 1 y hag >= min_hag
  - Método C (PointNet++): igual que B pero con pred de PointNet++

Algoritmo:
  1. Voxelizar la nube en grilla 3D
  2. Calcular densidad de puntos por voxel
  3. Suavizado gaussiano 3D del volumen de densidad
  4. Detectar semillas sobre la envolvente superior del volumen (ver nota)
  5. Crear marcadores 3D (voxel de copa por árbol)
  6. Watershed 3D sobre densidad invertida, con máscara density > 0
  7. Propagar etiquetas a puntos, filtrar segmentos pequeños

NOTA IMPORTANTE — Seeding 2D sobre envolvente vs. fill 3D volumétrico:

  El paso de seeding se ejecuta sobre una *envolvente superior del grid 3D*
  (para cada columna (x, y) se toma el voxel ocupado más alto), que es
  equivalente a un CHM derivado localmente del propio volumen y NO a un
  raster CHM externo. El verdadero aporte de este segmentador respecto a un
  watershed 2D clásico NO es el seeding sino el fill: el watershed opera
  sobre el grid 3D completo con máscara volumétrica, de modo que cada punto
  se asigna al voxel tridimensional que lo contiene — no a la proyección XY
  de su columna. Esto separa correctamente troncos adyacentes y puntos de
  sotobosque que caen bajo la proyección horizontal de una copa vecina,
  cosa que un watershed 2D puro no puede hacer.

  El seeding se deja en 2D a propósito: los máximos locales de densidad 3D
  caen en los troncos y ramas gruesas de las copas (no en los ápices),
  produciendo semillas que arruinarían la segmentación. La práctica
  estándar ITS (Dalponte-Coomes 2016) usa envolventes tipo CHM para
  seeding incluso cuando el fill es 3D, por la misma razón.

Parámetros justificados:
  - gaussian_sigma: calibrado por grid search independiente por flujo
    sobre el split val (ver evaluation/grid_search.py).
  - min_crown_radius_m: calibrado por grid search independiente por flujo
    sobre el split val.
  - CHM vectorizado: numpy en lugar de loop Python O(gx×gy).
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def voxelize(
    points: np.ndarray,
    voxel_size: float,
) -> tuple:
    """
    Voxeliza una nube de puntos en una grilla 3D.

    Args:
        points: (N, 3) coordenadas XYZ.
        voxel_size: Resolución del voxel en metros.

    Returns:
        density_grid: (gx, gy, gz) float32, densidad de puntos por voxel.
        point_to_voxel: (N, 3) int32, índice de voxel [i,j,k] por punto.
        origin: (3,) float64, coordenada del voxel [0,0,0].
        grid_shape: (3,) tuple con dimensiones de la grilla.
    """
    origin = points.min(axis=0)
    point_to_voxel = ((points - origin) / voxel_size).astype(np.int32)

    grid_shape = point_to_voxel.max(axis=0) + 1
    gx, gy, gz = grid_shape

    density_grid = np.zeros((gx, gy, gz), dtype=np.float32)
    np.add.at(
        density_grid,
        (point_to_voxel[:, 0], point_to_voxel[:, 1], point_to_voxel[:, 2]),
        1,
    )

    return density_grid, point_to_voxel, origin, tuple(grid_shape)


def watershed3d(
    points: np.ndarray,
    voxel_size: float = 0.1,
    min_tree_height: float = 2.0,
    min_points_per_tree: int = 50,
    gaussian_sigma: float = 0.5,
    min_crown_radius_m: float = 1.0,
) -> np.ndarray:
    """
    Segmenta árboles individuales mediante Watershed volumétrico 3D.

    Algoritmo detallado:
      1. Voxelizar nube -> density_grid (gx, gy, gz)
      2. Gaussian smoothing sobre el volumen
      3. Detectar semillas (una por árbol):
         a. Crear canopy height map 2D: para cada columna XY, el índice Z
            del voxel más alto con puntos (vectorizado numpy)
         b. peak_local_max sobre el CHM 2D con min_distance basado en
            min_crown_radius_m
         c. Filtrar semillas con altura < min_tree_height
      4. Crear marcadores 3D: voxel de copa por árbol
      5. Watershed sobre -density_smooth, con mask=density>0
      6. Propagar etiquetas de voxel a puntos originales
      7. Filtrar segmentos con < min_points_per_tree puntos

    Args:
        points: (N, 3) coordenadas de los puntos a segmentar. Se asume que la
            tercera coordenada es HAG (altura sobre el suelo) y NO Z absoluto:
            los pipelines construyen `np.column_stack([x, y, hag])` antes de
            llamar a esta función para que la voxelización sea invariante al
            relieve.
        voxel_size: Resolución del voxel en metros (default 0.1m).
        min_tree_height: Altura mínima para semilla (default 2.0m).
        min_points_per_tree: Mínimo de puntos por segmento (default 50).
        gaussian_sigma: Sigma del suavizado gaussiano (default 0.5).
            Valor calibrado por flujo mediante grid search sobre val set.
        min_crown_radius_m: Radio mínimo entre copas en metros (default 1.0m).
            Valor calibrado por flujo mediante grid search sobre val set.

    Returns:
        instance_ids: (N,) int32. >= 1 para árboles, 0 = sin asignación.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=np.int32)

    # --- 1. Voxelización ---
    density_grid, point_to_voxel, origin, (gx, gy, gz) = voxelize(
        points, voxel_size
    )

    # --- 2. Suavizado gaussiano ---
    density_smooth = gaussian_filter(
        density_grid.astype(np.float64), sigma=gaussian_sigma
    )

    # --- 3. Detectar semillas via CHM 2D (vectorizado) ---
    occupied = density_grid > 0

    # CHM vectorizado: índice Z del voxel más alto ocupado en cada columna XY.
    # Flip en eje Z para que argmax encuentre el voxel MÁS ALTO.
    # Columnas sin puntos quedan en 0.
    has_points = occupied.any(axis=2)
    flipped = occupied[:, :, ::-1]
    chm = np.where(
        has_points,
        (gz - 1) - np.argmax(flipped, axis=2),
        0,
    ).astype(np.int32)

    # CHM en metros para peak detection
    chm_m = chm.astype(np.float32) * voxel_size

    min_distance_voxels = max(1, int(min_crown_radius_m / voxel_size))

    # peak_local_max sobre el CHM 2D: `threshold_abs=min_tree_height`
    # (en metros) ya garantiza que las semillas retornadas correspondan a
    # columnas con top_z * voxel_size >= min_tree_height, por lo que no es
    # necesario un segundo filtro redundante sobre el índice de voxel.
    coords_2d = peak_local_max(
        chm_m,
        min_distance=min_distance_voxels,
        threshold_abs=min_tree_height,
        exclude_border=False,
    )

    if len(coords_2d) == 0:
        return np.zeros(len(points), dtype=np.int32)

    valid_seeds = [(int(ix), int(iy), int(chm[ix, iy])) for ix, iy in coords_2d]

    # --- 4. Crear marcadores 3D ---
    markers = np.zeros_like(density_grid, dtype=np.int32)
    for tree_id, (ix, iy, top_z) in enumerate(valid_seeds, start=1):
        markers[ix, iy, top_z] = tree_id

    # --- 5. Watershed ---
    mask = density_grid > 0
    labels = watershed(
        -density_smooth,
        markers=markers,
        mask=mask,
    )

    # --- 6. Propagar etiquetas a puntos ---
    instance_ids = labels[
        point_to_voxel[:, 0],
        point_to_voxel[:, 1],
        point_to_voxel[:, 2],
    ].astype(np.int32)

    # --- 7. Filtrar segmentos pequeños ---
    unique_ids, counts = np.unique(instance_ids, return_counts=True)
    small = unique_ids[(counts < min_points_per_tree) & (unique_ids > 0)]
    if len(small) > 0:
        instance_ids[np.isin(instance_ids, small)] = 0

    return instance_ids
