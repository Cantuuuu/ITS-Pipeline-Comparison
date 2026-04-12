"""
Flujo A — Baseline (sin preprocesamiento semántico).

Pipeline: Nube normalizada (excluyendo clases 0 y 3) -> Watershed 3D -> Instancias.

Este flujo sirve como línea base. Todos los puntos válidos (incluyendo suelo
y vegetación baja) entran al Watershed 3D. Esto es intencional — es el punto
de referencia sin filtrado semántico, para medir el valor añadido del
preprocesamiento en los Flujos B (RF) y C (PointNet++).

Parámetros del watershed: SIEMPRE se cargan de `grid_search_best_params.csv`.
Si no existe, el script falla con mensaje claro pidiendo correr primero
`python -m forest_its.evaluation.grid_search --methods baseline`.
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
from forest_its.evaluation.best_params import (
    load_best_watershed_params, MissingBestParamsError,
)


def run_baseline_single(las_path: Path, cfg, ws_params: dict) -> dict:
    """Ejecuta el pipeline baseline sobre un solo plot."""
    data = load_las(las_path)
    binary_labels = get_binary_labels(data["classification"])

    # Excluir clases 0 y 3 (label == -1)
    valid_mask = binary_labels != -1

    # Normalizar altura SOBRE LA NUBE COMPLETA (igual que los flujos B y C),
    # para que el DTM sea idéntico entre flujos y la única diferencia sea
    # qué puntos entran al watershed. Luego se indexa por valid_mask.
    hag_full = compute_hag(
        data["xyz"],
        classification=data["classification"],
        resolution=cfg.preprocessing.dtm_resolution,
        smooth_window=cfg.preprocessing.smooth_window,
        hag_min=cfg.preprocessing.hag_min,
        hag_max=cfg.preprocessing.hag_max,
    )

    xyz_valid = data["xyz"][valid_mask]
    hag = hag_full[valid_mask]

    # Watershed sobre TODOS los puntos válidos (sin filtro semántico).
    # Nota: se usa HAG en vez de Z absoluto para invarianza al relieve.
    points_for_ws = np.column_stack([xyz_valid[:, 0], xyz_valid[:, 1], hag])

    instance_ids_valid = watershed3d(
        points_for_ws,
        voxel_size=ws_params["voxel_size"],
        min_tree_height=ws_params["min_tree_height"],
        min_points_per_tree=ws_params["min_points_per_tree"],
        gaussian_sigma=ws_params["gaussian_sigma"],
        min_crown_radius_m=ws_params["min_crown_radius_m"],
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

    plot_name = f"{las_path.parent.name}__{las_path.stem}"
    return {
        "instance_ids": instance_ids,
        "plot": plot_name,
        "institution": las_path.parent.name,
        **inst_metrics,
    }


def run_baseline(cfg, split: str = "val"):
    """Ejecuta el pipeline baseline sobre los plots del split."""
    output_dir = Path(cfg.paths.output_dir)
    dataset_root = Path(cfg.paths.dataset_root)

    # Cargar parametros del watershed calibrados por grid search
    try:
        ws_params = load_best_watershed_params("baseline", cfg)
    except MissingBestParamsError as e:
        print(f"[ERROR] {e}")
        return
    print(f"Watershed params (grid search):")
    for k, v in ws_params.items():
        print(f"  {k}: {v}")

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
            result = run_baseline_single(las_path, cfg, ws_params)
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
