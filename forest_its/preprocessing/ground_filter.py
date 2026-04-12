"""
Extracción de DTM (Digital Terrain Model) para normalización de altura.

Estrategia: para cada celda de la grilla horizontal de resolución
configurable (0.5 m por defecto), se toma la mediana de Z sobre los
puntos clasificados como suelo (clase 2 de FOR-instance). FOR-instance
proporciona anotación humana de suelo de alta calidad, por lo que la
mediana sobre puntos de clase 2 es más robusta y precisa que un filtro
morfológico ciego (p. ej. percentil 5 sobre todos los puntos).

Celdas vacías se rellenan por interpolación bilineal (scipy.interpolate.griddata),
con fallback 'nearest' para bordes. Suavizado final con un filtro
gaussiano de sigma equivalente a una ventana 5×5 (σ ≈ 2.5 celdas) para
reducir artefactos en transiciones de celda.

Si la nube no aporta puntos de clase 2 (caso no esperado en FOR-instance),
se lanza ValueError explícitamente: el método requiere ground truth de
suelo y la rama de fallback morfológico ha sido eliminada.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator, griddata


def extract_dtm(
    xyz: np.ndarray,
    classification: np.ndarray = None,
    resolution: float = 0.5,
    smooth_window: int = 5,
) -> tuple:
    """
    Extrae el DTM de una nube de puntos de bosque.

    Para cada celda de la grilla se toma la mediana de Z sobre los puntos
    con Classification==2 (Terrain). Celdas sin puntos de suelo se
    rellenan por interpolación bilineal usando
    scipy.interpolate.griddata(method='linear'), con fallback 'nearest'
    para celdas en los bordes. El DTM resultante se suaviza con un
    filtro gaussiano de sigma equivalente a una ventana smooth_window×
    smooth_window (σ = smooth_window / 2).

    Args:
        xyz: (N, 3) coordenadas 3D de la nube de puntos.
        classification: (N,) etiquetas de clasificación. Obligatorio: el
            DTM se construye exclusivamente a partir de puntos de
            clase 2 (terreno) anotados.
        resolution: Tamaño de celda en metros (default 0.5m).
        smooth_window: Tamaño nominal del kernel de suavizado (default 5).
            Se traduce a un sigma del filtro gaussiano como
            σ = smooth_window / 2.

    Returns:
        dtm_grid: (ny, nx) array float64 con elevaciones del terreno.
        x_edges: (nx+1,) bordes de grilla en X.
        y_edges: (ny+1,) bordes de grilla en Y.
        x_centers: (nx,) centros de celda en X.
        y_centers: (ny,) centros de celda en Y.

    Raises:
        ValueError: si `classification` es None o no contiene puntos de
            clase 2. Este método requiere ground truth de suelo (clase 2
            de FOR-instance) y no implementa fallback morfológico.
    """
    if classification is None or (classification == 2).sum() == 0:
        raise ValueError(
            "extract_dtm requiere puntos clasificados como suelo "
            "(clase 2 de FOR-instance); ningún plot del benchmark "
            "carece de esta clase, así que un input sin clase 2 indica "
            "un bug en el cargador o un dataset distinto."
        )
    xyz_ground = xyz[classification == 2]

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
        # Mediana sobre puntos de clase 2 (terreno anotado): robusta a
        # outliers de baja altura como hojarasca o vegetación rasante.
        dtm_grid[cy, cx] = np.median(cell_z)

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

    # Suavizar con un filtro gaussiano cuyo sigma equivale a una ventana
    # smooth_window x smooth_window (σ = smooth_window / 2). Convención
    # más común en la literatura de DTM forestal que el filtro media.
    dtm_grid = gaussian_filter(
        dtm_grid.astype(np.float64),
        sigma=smooth_window / 2.0,
    )

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
