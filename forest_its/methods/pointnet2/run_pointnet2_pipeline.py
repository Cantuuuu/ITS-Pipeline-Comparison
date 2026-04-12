"""
Pipeline Flujo C: PointNet++ MSG semántico + Watershed 3D.

El pipeline está dividido en dos etapas para permitir grid search por flujo:

  Stage 'semantic'  — corre PointNet++ y guarda semantic_pred en
                      output/predictions/pointnet2/*_instances.npz.
                      Requiere: modelo PointNet++ entrenado (train_pointnet2.py).
                      Produce: semantic_pred por plot.

  Stage 'instance'  — lee semantic_pred, corre watershed 3D con los parametros
                      óptimos del grid search (output/results/grid_search_best_params.csv),
                      computa metricas y guarda el CSV final.
                      Requiere: grid search corrido (grid_search.py).
                      Produce: instance_ids + CSV de metricas.

Flujo de trabajo completo:
  1. python -m forest_its.methods.pointnet2.train_pointnet2
  2. python -m forest_its.methods.pointnet2.run_pointnet2_pipeline --stage semantic --split val
  3. python -m forest_its.evaluation.grid_search --methods pointnet2
  4. python -m forest_its.methods.pointnet2.run_pointnet2_pipeline --stage instance --split val
  5. python -m forest_its.methods.pointnet2.run_pointnet2_pipeline --stage instance --split test
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
from forest_its.methods.pointnet2.predict_pointnet2 import predict_plot_pointnet2, load_model
from forest_its.segmentation.watershed3d import watershed3d
from forest_its.evaluation.semantic_metrics import compute_semantic_metrics
from forest_its.evaluation.instance_metrics import compute_instance_metrics_plot
from forest_its.evaluation.best_params import (
    load_best_watershed_params, MissingBestParamsError,
)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─────────────────────────────────────────────
# Stage: semantic (corre PointNet++, guarda preds)
# ─────────────────────────────────────────────

def _run_semantic_single(
    las_path: Path,
    model,
    cfg,
    output_dir: Path,
    device,
    logger: logging.Logger,
) -> dict:
    """Corre PointNet++ sobre un plot y guarda semantic_pred en disco."""
    plot_name = f"{las_path.parent.name}__{las_path.stem}"

    data = load_las(las_path)
    data["_plot_stem"] = plot_name

    process_plot(
        data,
        resolution_dtm=cfg.preprocessing.dtm_resolution,
        smooth_window=cfg.preprocessing.smooth_window,
        hag_min=cfg.preprocessing.hag_min,
        hag_max=cfg.preprocessing.hag_max,
    )

    pred_binary, pred_proba = predict_plot_pointnet2(data, model, cfg, device)

    # Guardar predicciones (sin instancias todavia)
    pred_dir = output_dir / "predictions" / "pointnet2"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        pred_dir / f"{plot_name}_instances.npz",
        semantic_pred=pred_binary,
    )

    # Metricas semanticas para log
    binary_labels = get_binary_labels(data["classification"])
    valid = binary_labels != -1
    sem_metrics = compute_semantic_metrics(
        binary_labels, pred_binary, mask_valid=valid,
    )
    logger.info(f"  {plot_name}: sem mIoU={sem_metrics['miou']:.4f}, "
                f"tree IoU={sem_metrics['iou_tree']:.4f}")
    return {
        "plot": plot_name,
        "institution": las_path.parent.name,
        **{f"sem_{k}": v for k, v in sem_metrics.items()
           if k not in ("confusion_matrix",)},
    }


# ─────────────────────────────────────────────
# Stage: instance (lee preds, corre watershed con best params)
# ─────────────────────────────────────────────

def _run_instance_single(
    las_path: Path,
    ws_params: dict,
    cfg,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    """Lee semantic_pred pre-calculada + corre watershed con best params."""
    plot_name = f"{las_path.parent.name}__{las_path.stem}"
    institution = las_path.parent.name

    pred_file = output_dir / "predictions" / "pointnet2" / f"{plot_name}_instances.npz"
    if not pred_file.exists():
        raise FileNotFoundError(
            f"No existe {pred_file}. "
            f"Corre primero: run_pointnet2_pipeline.py --stage semantic --split ..."
        )
    pred = np.load(pred_file)
    pred_binary = pred["semantic_pred"]

    # Cargar plot y HAG (necesario para filter_tree_points)
    data = load_las(las_path)
    process_plot(
        data,
        resolution_dtm=cfg.preprocessing.dtm_resolution,
        smooth_window=cfg.preprocessing.smooth_window,
        hag_min=cfg.preprocessing.hag_min,
        hag_max=cfg.preprocessing.hag_max,
    )

    # Filtrar puntos arbol
    tree_xyz, tree_mask = filter_tree_points(
        data["xyz"], data["hag"], pred_binary,
        min_hag=cfg.preprocessing.min_hag_tree_filter,
    )
    n_tree_pts = int(tree_mask.sum())
    logger.info(f"  {plot_name}: {n_tree_pts} tree points "
                f"({100.0 * n_tree_pts / len(data['xyz']):.1f}%)")

    # Watershed 3D con best params (HAG como eje Z — invarianza al relieve)
    if n_tree_pts > 0:
        tree_hag = data["hag"][tree_mask]
        points_for_ws = np.column_stack([
            tree_xyz[:, 0], tree_xyz[:, 1], tree_hag,
        ])
        instance_ids_tree = watershed3d(
            points_for_ws,
            voxel_size=ws_params["voxel_size"],
            min_tree_height=ws_params["min_tree_height"],
            min_points_per_tree=ws_params["min_points_per_tree"],
            gaussian_sigma=ws_params["gaussian_sigma"],
            min_crown_radius_m=ws_params["min_crown_radius_m"],
        )
    else:
        instance_ids_tree = np.zeros(0, dtype=np.int32)

    instance_ids = np.zeros(len(data["xyz"]), dtype=np.int32)
    if n_tree_pts > 0:
        instance_ids[tree_mask] = instance_ids_tree

    # Metricas
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

    # Guardar instance_ids junto con semantic_pred
    np.savez(
        pred_file,
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


def _resolve_plot_paths(cfg, dataset_root: Path, split: str, single_plot: str):
    if single_plot:
        return [dataset_root / single_plot]
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


def run_pointnet2_pipeline(cfg, stage: str, split: str = "val", single_plot: str = None):
    """Ejecuta el pipeline PointNet++ en la etapa indicada ('semantic' o 'instance')."""
    output_dir = Path(cfg.paths.output_dir)
    dataset_root = Path(cfg.paths.dataset_root)

    logger = _setup_logger(
        f"pointnet2_pipeline_{stage}",
        output_dir / "logs" / f"pointnet2_pipeline_{stage}_{split}.log",
    )

    plot_paths = _resolve_plot_paths(cfg, dataset_root, split, single_plot)
    logger.info(f"=== PointNet2 Pipeline [{stage}] — {split} set — {len(plot_paths)} plots ===")

    if stage == "semantic":
        if not _TORCH_AVAILABLE:
            logger.error("PyTorch no disponible. Instalar con: pip install torch")
            return

        import torch
        from forest_its.methods.pointnet2.device_utils import select_device
        device = select_device()
        logger.info(f"Device: {device}")

        try:
            model = load_model(cfg, device)
            logger.info("Modelo PointNet++ cargado.")
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        pred_dir = output_dir / "predictions" / "pointnet2"
        pred_dir.mkdir(parents=True, exist_ok=True)

        all_metrics = []
        t_start = time.time()
        for las_path in tqdm(plot_paths, desc=f"PointNet2 semantic ({split})"):
            # Saltar plots ya procesados (permite retomar tras crash de MPS)
            npz_path = pred_dir / f"{las_path.stem}_instances.npz"
            if npz_path.exists():
                logger.info(f"  {las_path.stem}: ya existe, saltando")
                # Recargar métricas del plot para incluirlas en el CSV
                data = load_las(las_path)
                binary_labels = get_binary_labels(data["classification"])
                valid = binary_labels != -1
                pred_binary = np.load(npz_path)["semantic_pred"]
                sem_metrics = compute_semantic_metrics(
                    binary_labels, pred_binary, mask_valid=valid,
                )
                all_metrics.append({
                    "plot": las_path.stem,
                    "institution": las_path.parent.name,
                    **{f"sem_{k}": v for k, v in sem_metrics.items()
                       if k not in ("confusion_matrix",)},
                })
                continue
            try:
                result = _run_semantic_single(
                    las_path, model, cfg, output_dir, device, logger,
                )
                all_metrics.append(result)
            except NotImplementedError:
                logger.error(
                    "predict_plot_pointnet2 no implementado aún. "
                    "Completar después del entrenamiento."
                )
                return
            except Exception as e:
                logger.error(f"  FAILED {las_path.stem}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        t_total = time.time() - t_start

        if all_metrics:
            df = pd.DataFrame(all_metrics)
            results_dir = output_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            csv_path = results_dir / f"pointnet2_semantic_{split}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"\nSemantic stage done ({len(df)} plots, {t_total:.0f}s)")
            logger.info(f"  Mean sem mIoU: {df['sem_miou'].mean():.4f}")
            logger.info(f"  Saved: {csv_path}")
        return

    if stage == "instance":
        # Requiere best params del grid search
        try:
            ws_params = load_best_watershed_params("pointnet2", cfg)
        except MissingBestParamsError as e:
            logger.error(str(e))
            return
        logger.info("Watershed params (grid search):")
        for k, v in ws_params.items():
            logger.info(f"  {k}: {v}")

        all_metrics = []
        t_start = time.time()
        for las_path in tqdm(plot_paths, desc=f"PointNet2 instance ({split})"):
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
        csv_path = results_dir / f"pointnet2_metrics_{split}.csv"
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
            logger.info(f"    Accuracy:   {df['sem_accuracy'].mean():.4f}")
            logger.info(f"    seg F1:     {df['sem_seg_f1'].mean():.4f}")
        logger.info(f"\nSaved: {csv_path}")
        return

    raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointNet++ + Watershed 3D pipeline")
    parser.add_argument("--stage", required=True, choices=["semantic", "instance"],
                        help="'semantic' corre PointNet++ y guarda preds; "
                             "'instance' corre watershed con best params del grid search")
    parser.add_argument("--split", default="val", choices=["val", "test", "dev"])
    parser.add_argument("--plot", default=None,
                        help="Single plot path relative to dataset root")
    args = parser.parse_args()

    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
    )
    run_pointnet2_pipeline(cfg, stage=args.stage, split=args.split, single_plot=args.plot)
