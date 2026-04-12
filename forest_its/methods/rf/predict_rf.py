"""
Inferencia del Random Forest para predicción semántica binaria.

Aplica el RF entrenado a plots completos, genera predicciones por punto,
y filtra puntos predichos como árbol para el Watershed posterior.
"""

import sys
import numpy as np
import joblib
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from forest_its.data.dataset import load_las, get_binary_labels
from forest_its.preprocessing.normalize_height import process_plot
from forest_its.preprocessing.features_rf import compute_features_for_plot


def predict_plot(
    las_data: dict,
    rf_model,
    cfg,
    output_dir: Path,
) -> tuple:
    """
    Aplica el RF entrenado a un plot completo.

    Proceso:
      1. compute_features_for_plot() -> (N_valid, 27) con cache
      2. rf_model.predict() y predict_proba() sobre puntos válidos
      3. Para puntos con label == -1: predicción = -1 (excluidos)

    Args:
        las_data: Dict de load_las() con 'hag' y '_plot_stem' ya añadidos.
        rf_model: RandomForestClassifier entrenado.
        cfg: Configuración.
        output_dir: Directorio base de output.

    Returns:
        pred_binary: (N,) int32 con valores {-1, 0, 1}. -1 = excluido.
        pred_proba: (N, 2) float32 probabilidades [P(no-tree), P(tree)].
            Puntos excluidos tienen [0, 0].
    """
    binary_labels = get_binary_labels(las_data["classification"])
    valid_mask = binary_labels != -1
    n_total = len(las_data["xyz"])

    # Features (con cache)
    features = compute_features_for_plot(las_data, cfg, output_dir)

    # Predicción
    pred_valid = rf_model.predict(features).astype(np.int32)
    proba_valid = rf_model.predict_proba(features).astype(np.float32)

    # Reconstruir para todos los puntos
    pred_binary = np.full(n_total, -1, dtype=np.int32)
    pred_binary[valid_mask] = pred_valid

    pred_proba = np.zeros((n_total, 2), dtype=np.float32)
    pred_proba[valid_mask] = proba_valid

    return pred_binary, pred_proba


def filter_tree_points(
    xyz: np.ndarray,
    hag: np.ndarray,
    pred_binary: np.ndarray,
    min_hag: float = 0.5,
) -> tuple:
    """
    Filtra los puntos predichos como árbol para pasarlos al Watershed.

    Aplica dos filtros:
      1. pred_binary == 1  (predicción RF = árbol)
      2. hag >= min_hag    (eliminar puntos rasantes al suelo que el RF
                            clasificó como árbol por error)

    Justificación del filtro de altura: en UAV LiDAR, puntos de vegetación
    baja (<0.5m) pueden ser clasificados como árbol por similitud geométrica
    con ramas bajas. Filtrarlos mejora la calidad de semillas del Watershed.

    Args:
        xyz: (N, 3) coordenadas de la nube completa.
        hag: (N,) HAG de la nube completa.
        pred_binary: (N,) predicciones {-1, 0, 1}.
        min_hag: HAG mínimo para considerar punto como árbol (default 0.5m).

    Returns:
        tree_xyz: (M, 3) solo puntos árbol filtrados.
        tree_mask: (N,) booleano sobre la nube original.
    """
    tree_mask = (pred_binary == 1) & (hag >= min_hag)
    tree_xyz = xyz[tree_mask]
    return tree_xyz, tree_mask
