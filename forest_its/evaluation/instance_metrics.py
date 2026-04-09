"""
Métricas de evaluación de instancias para segmentación de árboles individuales.

Implementa el protocolo de evaluación estándar usado por ForAINet
(Henrich et al., 2024) y SegmentAnyTree (Wielgosz et al., 2024) sobre el
benchmark FOR-instance: un árbol predicho es True Positive si su IoU 3D
con algún árbol GT es >= 0.5.

treeID == 0 se excluye del matching: representa puntos sin árbol asignado
(suelo, vegetación baja), NO es un árbol con ID 0.
"""

import numpy as np
from typing import Dict


def compute_iou_3d(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    IoU entre dos máscaras booleanas sobre la nube de puntos.

    En segmentación de instancias 3D, el IoU se calcula sobre conjuntos
    de puntos (no voxeles), estándar en FOR-instance.

    Args:
        pred_mask: (N,) booleano.
        gt_mask: (N,) booleano.

    Returns:
        IoU en [0, 1].
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def match_instances(
    pred_ids: np.ndarray,
    gt_ids: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Empareja instancias predichas con GT mediante greedy matching por IoU.

    Reglas (ForAINet, Henrich et al. 2024; SegmentAnyTree, Wielgosz et al. 2024):
      - Ignorar pred_id == 0 y gt_id == 0
      - Para cada par (pred, GT), calcular IoU 3D
      - Greedy matching por IoU descendente: cada pred y GT se usa a lo sumo una vez
      - Si IoU >= iou_threshold -> TP
      - GT sin match -> FN
      - Pred sin match -> FP

    Args:
        pred_ids: (N,) IDs de instancia predichos (0 = sin asignación).
        gt_ids: (N,) IDs de instancia ground truth (0 = sin árbol).
        iou_threshold: Umbral para TP (default 0.5).

    Returns:
        Diccionario con: TP, FP, FN, precision, recall, f1, coverage,
        n_gt_trees, n_pred_trees, mean_iou_matched.
    """
    pred_unique = np.unique(pred_ids)
    pred_unique = pred_unique[pred_unique > 0]
    gt_unique = np.unique(gt_ids)
    gt_unique = gt_unique[gt_unique > 0]

    n_pred = len(pred_unique)
    n_gt = len(gt_unique)

    empty_result = {
        "TP": 0, "FP": 0, "FN": 0,
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "coverage": 0.0,
        "n_gt_trees": n_gt, "n_pred_trees": n_pred,
        "mean_iou_matched": 0.0,
    }

    if n_gt == 0:
        empty_result["FP"] = n_pred
        return empty_result

    if n_pred == 0:
        empty_result["FN"] = n_gt
        return empty_result

    # Calcular matriz de IoU
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i, pid in enumerate(pred_unique):
        pred_mask = pred_ids == pid
        for j, gid in enumerate(gt_unique):
            gt_mask = gt_ids == gid
            iou_matrix[i, j] = compute_iou_3d(pred_mask, gt_mask)

    # Greedy matching por IoU descendente
    matched_pred = set()
    matched_gt = set()
    matched_ious = []

    flat_indices = np.argsort(-iou_matrix.ravel())
    for flat_idx in flat_indices:
        i, j = divmod(int(flat_idx), n_gt)
        iou_val = iou_matrix[i, j]
        if iou_val < iou_threshold:
            break
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
            matched_ious.append(iou_val)

    tp = len(matched_gt)
    fp = n_pred - len(matched_pred)
    fn = n_gt - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    coverage = tp / n_gt
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "coverage": float(coverage),
        "n_gt_trees": n_gt,
        "n_pred_trees": n_pred,
        "mean_iou_matched": mean_iou,
    }


def compute_instance_metrics_plot(
    pred_ids: np.ndarray,
    gt_ids: np.ndarray,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Wrapper que calcula métricas de instancia + over/under segmentation.

    Args:
        pred_ids: (N,) IDs de instancia predichos.
        gt_ids: (N,) IDs de instancia GT.
        iou_threshold: Umbral IoU.

    Returns:
        Dict con todas las métricas de match_instances más over_seg y under_seg.
    """
    metrics = match_instances(pred_ids, gt_ids, iou_threshold)

    n_gt = metrics["n_gt_trees"]
    metrics["over_seg"] = float(metrics["FP"] / n_gt) if n_gt > 0 else 0.0
    metrics["under_seg"] = float(metrics["FN"] / n_gt) if n_gt > 0 else 0.0

    return metrics
