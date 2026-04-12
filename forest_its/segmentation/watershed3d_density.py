"""
Watershed 3D con seeding basado en densidad de copa (variante del watershed3d.py).

La única diferencia respecto a watershed3d.py es el paso de seeding (paso 3):
en lugar de buscar picos en el CHM (mapa de altura máxima por columna XY),
se buscan picos en una superficie de densidad de copa: para cada columna (x,y)
se suma la densidad suavizada en una banda cercana al tope de la columna.

Motivación:
  En bosques boreales densos (ej. NIBIO en FOR-instance), las copas tienen
  alturas muy similares → el CHM es casi plano → peak_local_max encuentra
  muy pocas semillas (2-3 donde hay 20-37 árboles GT). Al usar densidad
  en la banda superior, cada tronco/copa genera un pico local de densidad
  incluso cuando la altura es uniforme, produciendo muchas más semillas.

Todo lo demás (voxelización, gaussian smoothing, watershed fill 3D,
propagación, filtrado) es idéntico a watershed3d.py.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from forest_its.segmentation.watershed3d import voxelize


def watershed3d_density(
    points: np.ndarray,
    voxel_size: float = 0.1,
    min_tree_height: float = 2.0,
    min_points_per_tree: int = 50,
    gaussian_sigma: float = 0.5,
    min_crown_radius_m: float = 1.0,
    top_band_m: float = 3.0,
) -> np.ndarray:
    """
    Segmenta árboles individuales mediante Watershed 3D con seeding por densidad.

    Idéntico a watershed3d() excepto en el paso de detección de semillas:
    en lugar del CHM (max Z por columna), se usa una superficie 2D donde
    cada pixel (x,y) contiene la suma de densidad suavizada en los top_band_m
    metros superiores de esa columna. Esto detecta concentraciones de puntos
    (troncos, ramas gruesas) incluso en canopias de altura uniforme.

    Args:
        points: (N, 3) coordenadas (X, Y, HAG).
        voxel_size: Resolución del voxel en metros.
        min_tree_height: Altura mínima para semilla.
        min_points_per_tree: Mínimo de puntos por segmento.
        gaussian_sigma: Sigma del suavizado gaussiano 3D.
        min_crown_radius_m: Radio mínimo entre copas para peak_local_max.
        top_band_m: Grosor de la banda superior para acumular densidad (metros).

    Returns:
        instance_ids: (N,) int32. >= 1 para árboles, 0 = sin asignación.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=np.int32)

    # --- 1. Voxelización ---
    density_grid, point_to_voxel, origin, (gx, gy, gz) = voxelize(
        points, voxel_size
    )

    # --- 2. Suavizado gaussiano 3D ---
    density_smooth = gaussian_filter(
        density_grid.astype(np.float64), sigma=gaussian_sigma
    )

    # --- 3. Seeding por densidad en banda superior (vectorizado) ---
    occupied = density_grid > 0
    has_points = occupied.any(axis=2)

    # CHM: índice Z del voxel más alto ocupado en cada columna XY
    flipped = occupied[:, :, ::-1]
    chm = np.where(
        has_points,
        (gz - 1) - np.argmax(flipped, axis=2),
        0,
    ).astype(np.int32)

    # Banda superior: para cada columna, sumar densidad suavizada
    # en [top_z - band_voxels, top_z]
    band_voxels = max(1, int(top_band_m / voxel_size))

    # Construir superficie de densidad 2D (vectorizado con numpy)
    # Cumsum a lo largo de Z para sumas rápidas de rangos
    cumsum_z = np.cumsum(density_smooth, axis=2)  # (gx, gy, gz)

    # Índices de columnas con puntos
    ix_valid, iy_valid = np.where(has_points)
    top_z_vals = chm[ix_valid, iy_valid]  # (M,)
    bottom_z_vals = np.maximum(0, top_z_vals - band_voxels)

    # Suma en banda [bottom_z, top_z] via cumsum
    top_sums = cumsum_z[ix_valid, iy_valid, top_z_vals]
    # Para bottom_z == 0, la suma es cumsum[top_z]; si no, restar cumsum[bottom_z - 1]
    bottom_sums = np.where(
        bottom_z_vals > 0,
        cumsum_z[ix_valid, iy_valid, bottom_z_vals - 1],
        0.0,
    )
    density_surface = np.zeros((gx, gy), dtype=np.float64)
    density_surface[ix_valid, iy_valid] = top_sums - bottom_sums

    # Suavizar la superficie 2D para acentuar picos
    density_surface = gaussian_filter(density_surface, sigma=gaussian_sigma)

    # Peak detection sobre la superficie de densidad
    min_distance_voxels = max(1, int(min_crown_radius_m / voxel_size))
    min_tree_height_voxels = int(min_tree_height / voxel_size)

    coords_2d = peak_local_max(
        density_surface,
        min_distance=min_distance_voxels,
        threshold_abs=0,  # umbral relativo, no absoluto en altura
        exclude_border=False,
    )

    if len(coords_2d) == 0:
        return np.zeros(len(points), dtype=np.int32)

    # Filtrar semillas donde la columna no alcanza min_tree_height
    valid_seeds = []
    for ix, iy in coords_2d:
        top_z = int(chm[ix, iy])
        if top_z * voxel_size >= min_tree_height:
            valid_seeds.append((ix, iy, top_z))

    if len(valid_seeds) == 0:
        return np.zeros(len(points), dtype=np.int32)

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
