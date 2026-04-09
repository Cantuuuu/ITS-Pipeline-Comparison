"""
Pipeline de normalización de altura para un plot completo.

Wrapper que combina extract_dtm + normalize_height de ground_filter.py
en una sola llamada. Incluye process_plot() que añade el campo 'hag'
al diccionario de datos del plot.
"""

import numpy as np
from pathlib import Path

from .ground_filter import extract_dtm, normalize_height


def compute_hag(
    xyz: np.ndarray,
    classification: np.ndarray = None,
    resolution: float = 0.5,
    smooth_window: int = 5,
    hag_min: float = 0.0,
    hag_max: float = 50.0,
) -> np.ndarray:
    """
    Pipeline completo: extrae DTM y calcula HAG en un solo paso.

    Args:
        xyz: (N, 3) coordenadas 3D.
        classification: (N,) etiquetas semánticas opcionales.
        resolution: Resolución de la grilla DTM en metros.
        smooth_window: Tamaño del kernel de suavizado del DTM.
        hag_min: Clamp mínimo del HAG.
        hag_max: Clamp máximo del HAG.

    Returns:
        hag: (N,) float32 con altura normalizada por punto.
    """
    dtm_grid, _, _, x_centers, y_centers = extract_dtm(
        xyz,
        classification=classification,
        resolution=resolution,
        smooth_window=smooth_window,
    )

    hag = normalize_height(
        xyz, dtm_grid, x_centers, y_centers,
        hag_min=hag_min, hag_max=hag_max,
    )

    return hag


def process_plot(
    las_data: dict,
    resolution_dtm: float = 0.5,
    smooth_window: int = 5,
    hag_min: float = 0.0,
    hag_max: float = 50.0,
) -> dict:
    """
    Pipeline completo de normalización para un plot.

    Recibe el dict de load_las(), retorna ese mismo dict con campo 'hag' añadido.
    Usa Classification==2 preferentemente para el DTM.
    Imprime estadísticas: rango de HAG, % de puntos con HAG>0, altura media de copa.

    Args:
        las_data: Diccionario retornado por load_las().
        resolution_dtm: Resolución del DTM en metros.
        smooth_window: Kernel de suavizado del DTM.
        hag_min: Clamp mínimo.
        hag_max: Clamp máximo.

    Returns:
        El mismo diccionario con campo 'hag' (N,) float32 añadido.
    """
    from .ground_filter import extract_dtm, normalize_height
    from forest_its.data.dataset import get_binary_labels

    xyz = las_data["xyz"]
    classification = las_data["classification"]

    dtm_grid, _, _, x_centers, y_centers = extract_dtm(
        xyz,
        classification=classification,
        resolution=resolution_dtm,
        smooth_window=smooth_window,
    )

    hag = normalize_height(
        xyz, dtm_grid, x_centers, y_centers,
        hag_min=hag_min, hag_max=hag_max,
    )

    las_data["hag"] = hag

    # Estadísticas
    binary = get_binary_labels(classification)
    valid = binary != -1
    tree = binary == 1

    print(f"  HAG range: [{hag.min():.2f}, {hag.max():.2f}] m")
    print(f"  Points with HAG > 0: {(hag[valid] > 0).sum()}/{valid.sum()} "
          f"({100.0 * (hag[valid] > 0).mean():.1f}%)")
    if tree.any():
        print(f"  Mean tree HAG: {hag[tree].mean():.2f} m")

    return las_data
