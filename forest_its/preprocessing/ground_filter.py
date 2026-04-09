"""
Extracción de DTM (Digital Terrain Model) para normalización de altura.

Estrategia de dos pasos:
  1. Si hay puntos con Classification==2 (Terrain), usarlos directamente
     como ground truth de suelo. Estos son los más confiables en FOR-instance.
  2. Fallback: filtro morfológico mínimo (percentil 5 de Z por celda de grilla).

Celdas vacías se rellenan por interpolación bilineal (scipy.interpolate.griddata),
con fallback 'nearest' para bordes. Suavizado final con uniform_filter(size=5)
para eliminar artefactos en transiciones de celda.

Este enfoque es simple pero efectivo para UAV LiDAR forestal con alta densidad
(~18k pts/m², Wielgosz et al. 2024).
"""

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.interpolate import RegularGridInterpolator, griddata


def extract_dtm(
    xyz: np.ndarray,
    classification: np.ndarray = None,
    resolution: float = 0.5,
    smooth_window: int = 5,
) -> tuple:
    """
    Extrae el DTM de una nube de puntos de bosque.

    Paso 1: si classification está disponible, usar puntos con
    Classification==2 (Terrain) directamente como ground truth de suelo.
    Paso 2 (fallback): filtro morfológico mínimo — percentil 5 de Z por celda.

    Celdas sin puntos se rellenan por interpolación bilineal usando
    scipy.interpolate.griddata(method='linear'), con fallback 'nearest'
    para celdas en los bordes.

    Args:
        xyz: (N, 3) coordenadas 3D de la nube de puntos.
        classification: (N,) etiquetas de clasificación opcionales.
            Si se provee, los puntos con valor 2 se usan como terreno.
        resolution: Tamaño de celda en metros (default 0.5m).
        smooth_window: Tamaño del kernel de suavizado (default 5).

    Returns:
        dtm_grid: (ny, nx) array float64 con elevaciones del terreno.
        x_edges: (nx+1,) bordes de grilla en X.
        y_edges: (ny+1,) bordes de grilla en Y.
        x_centers: (nx,) centros de celda en X.
        y_centers: (ny,) centros de celda en Y.
    """
    # Determinar puntos de suelo
    use_terrain_class = False
    if classification is not None:
        terrain_mask = classification == 2
        if terrain_mask.sum() > 100:
            xyz_ground = xyz[terrain_mask]
            use_terrain_class = True
        else:
            xyz_ground = xyz
    else:
        xyz_ground = xyz

    # Usar el extent de TODA la nube para la grilla (no solo los puntos de suelo)
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    # Crear bordes de la grilla
    x_edges = np.arange(x_min, x_max + resolution, resolution)
    y_edges = np.arange(y_min, y_max + resolution, resolution)

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    if nx < 1 or ny < 1:
        z_min = xyz[:, 2].min()
        x_centers = np.array([(x_min + x_max) / 2.0])
        y_centers = np.array([(y_min + y_max) / 2.0])
        x_edges = np.array([x_min, x_max])
        y_edges = np.array([y_min, y_max])
        return np.array([[z_min]]), x_edges, y_edges, x_centers, y_centers

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    # Asignar puntos de suelo a celdas
    ix = np.clip(((xyz_ground[:, 0] - x_min) / resolution).astype(int), 0, nx - 1)
    iy = np.clip(((xyz_ground[:, 1] - y_min) / resolution).astype(int), 0, ny - 1)

    # Calcular elevación del terreno por celda
    dtm_grid = np.full((ny, nx), np.nan, dtype=np.float64)
    cell_idx = iy * nx + ix

    for cell in np.unique(cell_idx):
        cy, cx = divmod(int(cell), nx)
        cell_z = xyz_ground[cell_idx == cell, 2]
        if use_terrain_class:
            # Con puntos de terreno GT, la mediana es más robusta
            dtm_grid[cy, cx] = np.median(cell_z)
        else:
            # Sin GT, percentil 5 como aproximación al suelo
            dtm_grid[cy, cx] = np.percentile(cell_z, 5)

    # Rellenar celdas vacías con interpolación bilineal
    valid_mask = ~np.isnan(dtm_grid)
    if valid_mask.any() and not valid_mask.all():
        # Coordenadas de celdas válidas e inválidas
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        valid_points = np.column_stack([yy[valid_mask], xx[valid_mask]])
        valid_values = dtm_grid[valid_mask]
        invalid_points = np.column_stack([yy[~valid_mask], xx[~valid_mask]])

        # Interpolar linealmente, fallback nearest para bordes
        filled = griddata(valid_points, valid_values, invalid_points,
                          method="linear")
        still_nan = np.isnan(filled)
        if still_nan.any():
            filled_nearest = griddata(valid_points, valid_values,
                                      invalid_points[still_nan],
                                      method="nearest")
            filled[still_nan] = filled_nearest

        dtm_grid[~valid_mask] = filled
    elif not valid_mask.any():
        dtm_grid[:] = xyz[:, 2].min()

    # Suavizar para eliminar artefactos
    dtm_grid = uniform_filter(dtm_grid.astype(np.float64), size=smooth_window)

    return dtm_grid, x_edges, y_edges, x_centers, y_centers


def normalize_height(
    xyz: np.ndarray,
    dtm_grid: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    hag_min: float = 0.0,
    hag_max: float = 50.0,
) -> np.ndarray:
    """
    Calcula HAG (Height Above Ground) para cada punto interpolando el DTM.

    HAG es el feature más discriminativo para clasificación semántica en
    contexto forestal (peso=0.21 según Bremer et al. 2023; Li et al. 2023).
    Se calcula como Z - DTM(x,y).

    Se clampea a [hag_min, hag_max] para eliminar valores negativos
    (puntos bajo tierra por ruido del sensor) y outliers extremos.

    Args:
        xyz: (N, 3) coordenadas de la nube de puntos.
        dtm_grid: (ny, nx) array de elevaciones del terreno.
        x_centers: (nx,) centros de celdas en X.
        y_centers: (ny,) centros de celdas en Y.
        hag_min: Valor mínimo de HAG (default 0.0).
        hag_max: Valor máximo de HAG (default 50.0).

    Returns:
        hag: (N,) float32 con altura sobre el terreno por punto.
    """
    interpolator = RegularGridInterpolator(
        (y_centers, x_centers),
        dtm_grid,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    # RegularGridInterpolator espera (y, x)
    query_points = np.column_stack([xyz[:, 1], xyz[:, 0]])
    terrain_z = interpolator(query_points)

    hag = xyz[:, 2] - terrain_z
    hag = np.clip(hag, hag_min, hag_max).astype(np.float32)

    return hag
