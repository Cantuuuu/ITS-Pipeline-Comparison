"""
Pipeline Flujo E: PointNet++ semántico (reutilizado) + Watershed 3D con seeding por densidad.

Análogo a rf_density pero reutiliza predicciones semánticas de PointNet++
en output/predictions/pointnet2/. No re-entrena ni re-predice.

Uso:
  python -m forest_its.methods.pointnet2_density.run_pointnet2_density_pipeline --stage instance --split val
  python -m forest_its.methods.pointnet2_density.run_pointnet2_density_pipeline --stage instance --split test
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
from forest_its.preprocessing.normalize_height import process_plot
from forest_its.methods.rf.predict_rf import filter_tree_points
from forest_its.segmentation.watershed3d_density import watershed3d_density
from forest_its.evaluation.semantic_metrics import compute_semantic_metrics
from forest_its.evaluation.instance_metrics import compute_instance_metrics_plot
from forest_its.evaluation.best_params import (
    load_best_watershed_params, MissingBestParamsError,
)


def _run_instance_single(
    las_path: Path,
    ws_params: dict,
    cfg,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    """Lee semantic_pred de PointNet++ + corre watershed_density con best params."""
    plot_name = f"{las_path.parent.name}__{las_path.stem}"
    institution = las_path.parent.name

    # Reutilizar predicciones semánticas de PointNet++
    pred_file = output_dir / "predictions" / "pointnet2" / f"{plot_name}_instances.npz"
    if not pred_file.exists():
        raise FileNotFoundError(
            f"No existe {pred_file}. "
            f"Corre primero: run_pointnet2_pipeline.py --stage semantic --split ..."
        )
    pred = np.load(pred_file)
    pred_binary = pred["semantic_pred"]

    # Cargar plot y HAG
    data = load_las(las_path)
    process_plot(
        data,
        resolution_dtm=cfg.preprocessing.dtm_resolution,
        smooth_window=cfg.preprocessing.smooth_window,
        hag_min=cfg.preprocessing.hag_min,
        hag_max=cfg.preprocessing.hag_max,
    )

    # Filtrar puntos árbol (mismo filtro que los demás pipelines)
    tree_xyz, tree_mask = filter_tree_points(
        data["xyz"], data["hag"], pred_binary,
        min_hag=cfg.preprocessing.min_hag_tree_filter,
    )
    n_tree_pts = int(tree_mask.sum())
    logger.info(f"  {plot_name}: {n_tree_pts} tree points "
                f"({100.0 * n_tree_pts / len(data['xyz']):.1f}%)")

    # Watershed 3D density con best params
    if n_tree_pts > 0:
        tree_hag = data["hag"][tree_mask]
        points_for_ws = np.column_stack([
            tree_xyz[:, 0], tree_xyz[:, 1], tree_hag,
        ])
        instance_ids_tree = watershed3d_density(
            points_for_ws,
            voxel_size=ws_params["voxel_size"],
            min_tree_height=ws_params["min_tree_height"],
            min_points_per_tree=ws_params["min_points_per_tree"],
            gaussian_sigma=ws_params["gaussian_sigma"],
            min_crown_radius_m=ws_params["min_crown_radius_m"],
            top_band_m=ws_params["top_band_m"],
        )
    else:
        instance_ids_tree = np.zeros(0, dtype=np.int32)

    instance_ids = np.zeros(len(data["xyz"]), dtype=np.int32)
    if n_tree_pts > 0:
        instance_ids[tree_mask] = instance_ids_tree

    # Métricas
    binary_labels = get_binary_labels(data["classification"])
    valid = binary_labels != -1
    sem_metrics = compute_semantic_metrics(
        binary_labels, pred_binary, mask_valid=valid,
    )
    inst_metrics = compute_instance_metrics_plot(
        instance_ids[valid],
        data["tree_id"][valid],
        iou_threshold=cfg.evaluation.iou_threshold,
    )
    logger.info(f"    Pred trees: {inst_metrics['n_pred_trees']}, "
                f"GT trees: {inst_metrics['n_gt_trees']}, "
                f"F1: {inst_metrics['f1']:.3f}")

    # Guardar instance_ids
    save_dir = output_dir / "predictions" / "pointnet2_density"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_dir / f"{plot_name}_instances.npz",
        instance_ids=instance_ids,
        semantic_pred=pred_binary,
    )

    return {
        "plot": plot_name,
        "institution": institution,
        **{f"sem_{k}": v for k, v in sem_metrics.items()
           if k not in ("confusion_matrix",)},
        **inst_metrics,
    }


def _setup_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


def _resolve_plot_paths(cfg, dataset_root: Path, split: str):
    dev_paths, test_paths = load_splits(dataset_root)
    if split == "val":
        _, plot_paths = get_train_val_split(
            dev_paths,
            val_fraction=cfg.data.val_split,
            random_state=cfg.data.random_state,
        )
        return plot_paths
    if split == "test":
        return test_paths
    return dev_paths


def run_pointnet2_density_pipeline(cfg, stage: str, split: str = "val"):
    """Ejecuta el pipeline PointNet+++Density Watershed."""
    output_dir = Path(cfg.paths.output_dir)
    dataset_root = Path(cfg.paths.dataset_root)

    logger = _setup_logger(
        f"pointnet2_density_pipeline_{stage}",
        output_dir / "logs" / f"pointnet2_density_pipeline_{stage}_{split}.log",
    )

    plot_paths = _resolve_plot_paths(cfg, dataset_root, split)
    logger.info(f"=== PointNet++ Density Pipeline [{stage}] — {split} set — {len(plot_paths)} plots ===")

    if stage == "instance":
        try:
            ws_params = load_best_watershed_params("pointnet2_density", cfg)
        except MissingBestParamsError as e:
            logger.error(str(e))
            return
        logger.info("Watershed density params (grid search):")
        for k, v in ws_params.items():
            logger.info(f"  {k}: {v}")

        all_metrics = []
        t_start = time.time()
        for las_path in tqdm(plot_paths, desc=f"PN++ density instance ({split})"):
            try:
                result = _run_instance_single(
                    las_path, ws_params, cfg, output_dir, logger,
                )
                all_metrics.append(result)
            except Exception as e:
                logger.error(f"  FAILED {las_path.stem}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        t_total = time.time() - t_start

        if not all_metrics:
            logger.error("No plots processed successfully.")
            return

        df = pd.DataFrame(all_metrics)
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / f"pointnet2_density_metrics_{split}.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Results — {split} set ({len(df)} plots, {t_total:.0f}s)")
        logger.info(f"{'=' * 70}")
        logger.info(f"\n  Instance metrics (mean):")
        logger.info(f"    Precision:  {df['precision'].mean():.4f}")
        logger.info(f"    Recall:     {df['recall'].mean():.4f}")
        logger.info(f"    F1:         {df['f1'].mean():.4f}")
        logger.info(f"    Coverage:   {df['coverage'].mean():.4f}")
        logger.info(f"    GT trees:   {df['n_gt_trees'].sum()}")
        logger.info(f"    Pred trees: {df['n_pred_trees'].sum()}")
        if "sem_miou" in df.columns:
            logger.info(f"\n  Semantic metrics (mean):")
            logger.info(f"    mIoU:       {df['sem_miou'].mean():.4f}")
        logger.info(f"\nSaved: {csv_path}")
        return

    raise ValueError(f"Unknown stage: {stage}. pointnet2_density only supports 'instance'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PointNet++ (reused) + Watershed 3D Density Seeding pipeline"
    )
    parser.add_argument("--stage", required=True, choices=["instance"],
                        help="'instance' corre watershed density con best params")
    parser.add_argument("--split", default="val", choices=["val", "test", "dev"])
    args = parser.parse_args()

    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
    )
    run_pointnet2_density_pipeline(cfg, stage=args.stage, split=args.split)
