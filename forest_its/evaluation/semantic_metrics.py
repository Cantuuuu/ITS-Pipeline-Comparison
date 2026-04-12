"""
Métricas de evaluación semántica para los Métodos B (RF) y C (PointNet++).

Evalúa la calidad del preprocesamiento semántico binario (árbol vs no-árbol)
comparando las predicciones con el ground truth derivado de las anotaciones
oficiales de FOR-instance (Puliti et al., 2023).

Solo se evalúa sobre puntos con label != -1 (excluyendo Unclassified y
Out-points), ya que estos no tienen ground truth confiable.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
)


def compute_semantic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask_valid: np.ndarray = None,
) -> dict:
    """
    Métricas de segmentación semántica binaria (árbol/no-árbol).

    Solo evalúa sobre puntos donde mask_valid==True (excluye label==-1).
    Si mask_valid es None, se asume que ya fueron filtrados.

    Args:
        y_true: (N,) etiquetas ground truth {0, 1} (o {-1, 0, 1} si mask_valid).
        y_pred: (N,) etiquetas predichas {0, 1} (o {-1, 0, 1}).
        mask_valid: (N,) booleano opcional. Si provisto, filtra ambos arrays.

    Returns:
        Diccionario con:
          accuracy, iou_tree, iou_notree, miou,
          seg_f1 (= sklearn f1_score con pos_label=1, average='binary';
          es la única F1 que aparece en las tablas del paper),
          confusion_matrix (2x2), n_points_evaluated.
    """
    if mask_valid is not None:
        y_true = y_true[mask_valid]
        y_pred = y_pred[mask_valid]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # IoU por clase: TP_i / (TP_i + FP_i + FN_i)
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = intersection / np.maximum(union, 1).astype(float)
    iou_notree = float(iou[0])
    iou_tree = float(iou[1])
    miou = float(iou.mean())

    acc = float(accuracy_score(y_true, y_pred))
    seg_f1 = float(f1_score(
        y_true, y_pred,
        pos_label=1, labels=[0, 1], average="binary",
    ))

    return {
        "accuracy": acc,
        "iou_tree": iou_tree,
        "iou_notree": iou_notree,
        "miou": miou,
        "seg_f1": seg_f1,
        "confusion_matrix": cm,
        "n_points_evaluated": int(len(y_true)),
    }
