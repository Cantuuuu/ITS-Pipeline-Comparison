"""
Método A — Baseline (sin preprocesamiento semántico).

Pipeline: Nube normalizada (excluyendo clases 0 y 3) -> Watershed 3D -> Instancias.

Este método sirve como línea base. Todos los puntos válidos (incluyendo suelo
y vegetación baja) entran al Watershed 3D. Esto es intencional — es el punto
de referencia sin filtrado semántico, para medir el valor añadido del
preprocesamiento en los Métodos B y C.
"""

import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from forest_its.data.dataset import load_las, get_binary_labels, load_splits
from forest_its.data.splits import get_train_val_split
from forest_its.preprocessing.normalize_height import compute_hag
from forest_its.segmentation.watershed3d import watershed3d
from forest_its.evaluation.instance_metrics import compute_instance_metrics_plot


def run_baseline_single(las_path: Path, cfg) -> dict:
    """Ejecuta el pipeline baseline sobre un solo plot."""
    data = load_las(las_path)
    binary_labels = get_binary_labels(data["classification"])

    # Excluir clases 0 y 3 (label == -1)
    valid_mask = binary_labels != -1
    xyz_valid = data["xyz"][valid_mask]

    # Normalizar altura
    hag = compute_hag(
        xyz_valid,
        classification=data["classification"][valid_mask],
        resolution=cfg.preprocessing.dtm_resolution,
        smooth_window=cfg.preprocessing.smooth_window,
        hag_min=cfg.preprocessing.hag_min,
        hag_max=cfg.preprocessing.hag_max,
    )

    # Watershed sobre TODOS los puntos válidos (sin filtro semántico)
    points_for_ws = np.column_stack([xyz_valid[:, 0], xyz_valid[:, 1], hag])

    instance_ids_valid = watershed3d(
        points_for_ws,
        voxel_size=cfg.watershed.voxel_size,
        min_tree_height=cfg.watershed.min_tree_height,
        min_points_per_tree=cfg.watershed.min_points_per_tree,
        gaussian_sigma=cfg.watershed.gaussian_sigma,
        min_crown_radius_m=cfg.watershed.min_crown_radius_m,
    )

    # Reconstruir IDs para la nube completa
    instance_ids = np.zeros(len(data["xyz"]), dtype=np.int32)
    instance_ids[valid_mask] = instance_ids_valid

    # Métricas de instancia
    inst_metrics = compute_instance_metrics_plot(
        instance_ids[valid_mask],
        data["tree_id"][valid_mask],
        iou_threshold=cfg.evaluation.iou_threshold,
    )

    return {
        "instance_ids": instance_ids,
        "plot": las_path.stem,
        "institution": las_path.parent.name,
        **inst_metrics,
    }


def run_baseline(cfg, split: str = "val"):
    """Ejecuta el pipeline baseline sobre los plots del split."""
    output_dir = Path(cfg.paths.output_dir)
    dataset_root = Path(cfg.paths.dataset_root)

    dev_paths, test_paths = load_splits(dataset_root)
    if split == "val":
        _, plot_paths = get_train_val_split(
            dev_paths, val_fraction=cfg.data.val_split,
            random_state=cfg.data.random_state,
        )
    elif split == "test":
        plot_paths = test_paths
    else:
        plot_paths = dev_paths

    pred_dir = output_dir / "predictions" / "baseline"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Baseline — {split} set — {len(plot_paths)} plots ===")

    all_metrics = []
    for las_path in tqdm(plot_paths, desc=f"Baseline ({split})"):
        try:
            result = run_baseline_single(las_path, cfg)
            np.savez(
                pred_dir / f"{result['plot']}_instances.npz",
                instance_ids=result["instance_ids"],
            )
            all_metrics.append({k: v for k, v in result.items()
                               if k != "instance_ids"})
            print(f"  {result['plot']}: F1={result['f1']:.3f} "
                  f"(GT={result['n_gt_trees']}, Pred={result['n_pred_trees']})")
        except Exception as e:
            print(f"  FAILED {las_path.stem}: {e}")
            continue

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / f"baseline_metrics_{split}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nMean F1: {df['f1'].mean():.4f}")
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test", "dev"])
    args = parser.parse_args()

    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
    )
    run_baseline(cfg, split=args.split)
