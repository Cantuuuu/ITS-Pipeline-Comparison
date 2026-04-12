"""
Diagnóstico de cobertura KNN para CULS plot_3_annotated.

Reporta:
  1. Cuántos puntos asignados por KNN (sin cobertura directa)
  2. Clase predicha KNN vs. etiqueta real (Classification)
  3. Posición espacial de esos puntos (HAG, borde vs interior, XY)
"""

import sys
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from forest_its.data.dataset import load_las, get_binary_labels
from forest_its.preprocessing.normalize_height import process_plot
from forest_its.methods.pointnet2.predict_pointnet2 import load_model

import torch
from torch.amp import autocast

from forest_its.methods.pointnet2.device_utils import select_device


def run_diagnostic():
    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
    )
    device = select_device()
    print(f"Device: {device}")

    las_path = Path(cfg.paths.dataset_root) / "CULS" / "plot_3_annotated.las"
    print(f"Loading: {las_path}")

    data = load_las(las_path)
    data["_plot_stem"] = "plot_3_annotated"
    process_plot(
        data,
        resolution_dtm=cfg.preprocessing.dtm_resolution,
        smooth_window=cfg.preprocessing.smooth_window,
        hag_min=cfg.preprocessing.hag_min,
        hag_max=cfg.preprocessing.hag_max,
    )

    model = load_model(cfg, device)

    # ── replicate predict_plot_pointnet2 but save covered_mask ────────────────
    xyz_all = data["xyz"].astype(np.float32)
    hag_all = data["hag"].astype(np.float32)
    intensity_all = data["intensity"].astype(np.float32)
    classification = data["classification"]
    N = len(xyz_all)

    num_points = int(cfg.pointnet2.num_points)
    hag_max = float(cfg.preprocessing.hag_max)

    binary_labels = get_binary_labels(classification)
    valid_mask = binary_labels != -1
    xy_center = xyz_all[valid_mask, :2].mean(axis=0) if valid_mask.any() else xyz_all[:, :2].mean(axis=0)

    xyz_c = xyz_all.copy()
    xyz_c[:, 0] -= xy_center[0]
    xyz_c[:, 1] -= xy_center[1]

    hag_norm_all = np.clip(hag_all / hag_max, 0.0, 1.0)
    n_passes = min(1000, max(50, int(N / num_points * 3)))
    print(f"N={N}, n_passes={n_passes}")

    prob_sum = np.zeros((N, 2), dtype=np.float32)
    prob_count = np.zeros(N, dtype=np.int32)

    _INFER_BATCH = 16
    model.eval()
    with torch.no_grad():
        done = 0
        while done < n_passes:
            b_size = min(_INFER_BATCH, n_passes - done)
            b_xyz, b_feat, b_idx = [], [], []
            for _ in range(b_size):
                if N >= num_points:
                    choice = np.random.choice(N, num_points, replace=False)
                else:
                    choice = np.random.choice(N, num_points, replace=True)
                xyz_s = xyz_c[choice].copy()
                centroid = xyz_s.mean(axis=0)
                xyz_s -= centroid
                radius = float(np.max(np.sqrt((xyz_s ** 2).sum(axis=1)))) + 1e-8
                xyz_s /= radius
                hag_s = hag_norm_all[choice]
                int_s = intensity_all[choice]
                b_xyz.append(xyz_s)
                b_feat.append(np.column_stack([hag_s, int_s]).astype(np.float32))
                b_idx.append(choice)

            xyz_t = torch.from_numpy(np.stack(b_xyz)).to(device)
            feat_t = torch.from_numpy(np.stack(b_feat)).to(device)
            with autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=cfg.pointnet2.mixed_precision,
            ):
                logits = model(xyz_t, feat_t)
            probs_batch = torch.exp(logits).float().cpu().numpy()
            for indices, probs in zip(b_idx, probs_batch):
                np.add.at(prob_sum, indices, probs)
                np.add.at(prob_count, indices, 1)
            done += b_size

    covered_mask = prob_count > 0
    uncovered_mask = ~covered_mask
    n_uncovered = uncovered_mask.sum()

    print(f"\n{'='*60}")
    print(f"1. COBERTURA DIRECTA")
    print(f"{'='*60}")
    print(f"   Total puntos:       {N:,}")
    print(f"   Con cobertura:      {covered_mask.sum():,} ({100.*covered_mask.sum()/N:.1f}%)")
    print(f"   Sin cobertura (KNN):{n_uncovered:,} ({100.*n_uncovered/N:.1f}%)")

    # ── Normalizar cubiertos ────────────────────────────────────────────────
    pred_proba = np.zeros((N, 2), dtype=np.float32)
    pred_proba[covered_mask] = prob_sum[covered_mask] / prob_count[covered_mask, None]

    # ── KNN para no cubiertos ────────────────────────────────────────────────
    covered_xyz = xyz_c[covered_mask]
    covered_proba = pred_proba[covered_mask]
    uncovered_xyz = xyz_c[uncovered_mask]

    tree = cKDTree(covered_xyz)
    dists, idxs = tree.query(uncovered_xyz, k=3)
    weights = np.where(dists == 0, 1.0, 1.0 / (dists + 1e-8)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    knn_proba = np.einsum("nk,nkc->nc", weights, covered_proba[idxs])
    pred_proba[uncovered_mask] = knn_proba

    knn_pred = knn_proba.argmax(axis=1)       # 0=no-tree, 1=tree
    knn_gt_binary = binary_labels[uncovered_mask]   # -1, 0, 1
    knn_gt_raw = classification[uncovered_mask]      # raw Classification field

    print(f"\n{'='*60}")
    print(f"2. CLASE KNN vs. ETIQUETA REAL")
    print(f"{'='*60}")
    # Sólo puntos válidos (no excluidos)
    valid_unc = knn_gt_binary != -1
    print(f"   Puntos KNN válidos (clasificables): {valid_unc.sum():,} / {n_uncovered:,}")
    print(f"   Puntos KNN excluidos (Unclassified/Out): {(~valid_unc).sum():,}")

    knn_pred_valid = knn_pred[valid_unc]
    knn_gt_valid = knn_gt_binary[valid_unc]

    tp = ((knn_pred_valid == 1) & (knn_gt_valid == 1)).sum()
    fp = ((knn_pred_valid == 1) & (knn_gt_valid == 0)).sum()
    fn = ((knn_pred_valid == 0) & (knn_gt_valid == 1)).sum()
    tn = ((knn_pred_valid == 0) & (knn_gt_valid == 0)).sum()
    print(f"\n   Confusion matrix (KNN-assigned points only):")
    print(f"              Pred=tree  Pred=no-tree")
    print(f"   GT=tree    {tp:8,}   {fn:8,}   (GT tree total: {tp+fn:,})")
    print(f"   GT=no-tree {fp:8,}   {tn:8,}   (GT no-tree total: {fp+tn:,})")
    if (tp + fp) > 0:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"\n   KNN sem. precision: {prec:.3f}  recall: {rec:.3f}")

    # Raw class breakdown
    print(f"\n   Raw Classification of uncovered points:")
    CLASS_NAMES = {0: "Unclassified", 1: "Low-veg", 2: "Terrain",
                   3: "Out-points", 4: "Stem", 5: "Live-branches", 6: "Woody-branches"}
    for cls_val, cls_name in CLASS_NAMES.items():
        cnt = (knn_gt_raw == cls_val).sum()
        if cnt > 0:
            pred_tree_pct = 100. * knn_pred[knn_gt_raw == cls_val].mean()
            print(f"     Class {cls_val} ({cls_name:16s}): {cnt:8,} pts  KNN pred tree: {pred_tree_pct:.1f}%")

    print(f"\n{'='*60}")
    print(f"3. POSICIÓN ESPACIAL DE PUNTOS SIN COBERTURA")
    print(f"{'='*60}")

    unc_xyz = xyz_all[uncovered_mask]   # coordenadas originales (no centradas)
    unc_hag = hag_all[uncovered_mask]

    # HAG distribution
    print(f"\n   HAG distribution (height above ground):")
    hag_bins = [0, 0.5, 2, 5, 10, 20, 50]
    for lo, hi in zip(hag_bins[:-1], hag_bins[1:]):
        cnt = ((unc_hag >= lo) & (unc_hag < hi)).sum()
        pct = 100. * cnt / n_uncovered
        label = f"{lo}-{hi}m"
        print(f"     {label:10s}: {cnt:8,} ({pct:.1f}%)")

    # Borde vs interior: percentile-based bounding box
    xy_all = xyz_all[:, :2]
    xmin, xmax = xy_all[:, 0].min(), xy_all[:, 0].max()
    ymin, ymax = xy_all[:, 1].min(), xy_all[:, 1].max()
    dx = xmax - xmin
    dy = ymax - ymin
    margin = 0.1   # 10% del rango = borde

    unc_x = unc_xyz[:, 0]
    unc_y = unc_xyz[:, 1]
    border_mask = (
        (unc_x < xmin + margin * dx) | (unc_x > xmax - margin * dx) |
        (unc_y < ymin + margin * dy) | (unc_y > ymax - margin * dy)
    )
    print(f"\n   Plot XY extent: X=[{xmin:.1f}, {xmax:.1f}] ({dx:.1f}m)  Y=[{ymin:.1f}, {ymax:.1f}] ({dy:.1f}m)")
    print(f"   Border margin: 10% = {margin*dx:.1f}m (X), {margin*dy:.1f}m (Y)")
    print(f"   Borde (10%):   {border_mask.sum():,} ({100.*border_mask.sum()/n_uncovered:.1f}%)")
    print(f"   Interior:      {(~border_mask).sum():,} ({100.*(~border_mask).sum()/n_uncovered:.1f}%)")

    # Interior — are they clumped or spread?
    interior_unc = uncovered_mask.copy()
    interior_unc[uncovered_mask] &= ~border_mask
    interior_xyz = xyz_all[interior_unc]
    covered_only = covered_mask.copy()

    # Nearest-covered distance for uncovered
    tree_xy = cKDTree(xyz_all[covered_mask, :2])
    nn_dist, _ = tree_xy.query(unc_xyz[:, :2], k=1)
    print(f"\n   Distance to nearest covered neighbor (XY):")
    for lo, hi in [(0,0.5),(0.5,1),(1,2),(2,5),(5,100)]:
        cnt = ((nn_dist >= lo) & (nn_dist < hi)).sum()
        pct = 100. * cnt / n_uncovered
        print(f"     {lo:.1f}–{hi:.1f}m: {cnt:8,} ({pct:.1f}%)")

    print(f"\n   Mean distance to nearest covered neighbor: {nn_dist.mean():.3f}m")
    print(f"   Max  distance to nearest covered neighbor: {nn_dist.max():.3f}m")

    print(f"\nDone.")


if __name__ == "__main__":
    run_diagnostic()
