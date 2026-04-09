"""
Pipeline completo Método B: RF semántico + Watershed 3D.

Uso:
  python -m forest_its.methods.rf.run_rf_pipeline --split val
  python -m forest_its.methods.rf.run_rf_pipeline --split test
  python -m forest_its.methods.rf.run_rf_pipeline --plot CULS/plot_1_annotated.las

Flujo por plot:
  1. load_las() -> process_plot() -> HAG
  2. compute_features_for_plot() -> features con cache
  3. predict_plot() -> pred_binary, pred_proba
  4. filter_tree_points() -> tree_xyz, tree_mask
  5. watershed3d(tree_xyz) -> instance_ids_trees
  6. Mapear de vuelta a la nube original
  7. Calcular métricas semánticas y de instancia
  8. Guardar resultados
"""

import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from forest_its.data.dataset import load_las, get_binary_labels, load_splits
from forest_its.data.splits import get_train_val_split
from forest_its.preprocessing.normalize_height import process_plot, compute_hag
from forest_its.methods.rf.predict_rf import predict_plot, filter_tree_points
from forest_its.segmentation.watershed3d import watershed3d
from forest_its.evaluation.semantic_metrics import compute_semantic_metrics
from forest_its.evaluation.instance_metrics import compute_instance_metrics_plot


def _process_single_plot(
    las_path: Path,
    rf_model,
    cfg,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    """Procesa un plot completo: RF prediction + Watershed + metrics."""
    plot_name = las_path.stem
    institution = las_path.parent.name

    # 1. Cargar y normalizar
    data = load_las(las_path)
    data["_plot_stem"] = plot_name

    process_plot(
        data,
        resolution_dtm=cfg.preprocessing.dtm_resolution,
        smooth_window=cfg.preprocessing.smooth_window,
        hag_min=cfg.preprocessing.hag_min,
        hag_max=cfg.preprocessing.hag_max,
    )

    # 2-3. Predicción semántica
    pred_binary, pred_proba = predict_plot(data, rf_model, cfg, output_dir)

    # 4. Filtrar puntos árbol
    tree_xyz, tree_mask = filter_tree_points(
        data["xyz"], data["hag"], pred_binary,
        min_hag=cfg.preprocessing.min_hag_tree_filter,
    )

    n_tree_pts = tree_mask.sum()
    logger.info(f"  {plot_name}: {n_tree_pts} tree points "
                f"({100.0 * n_tree_pts / len(data['xyz']):.1f}%)")

    # 5. Watershed 3D
    if n_tree_pts > 0:
        # Usar HAG como Z para el watershed
        tree_hag = data["hag"][tree_mask]
        points_for_ws = np.column_stack([
            tree_xyz[:, 0], tree_xyz[:, 1], tree_hag,
        ])

        instance_ids_tree = watershed3d(
            points_for_ws,
            voxel_size=cfg.watershed.voxel_size,
            min_tree_height=cfg.watershed.min_tree_height,
            min_points_per_tree=cfg.watershed.min_points_per_tree,
            gaussian_sigma=cfg.watershed.gaussian_sigma,
            min_crown_radius_m=cfg.watershed.min_crown_radius_m,
        )
    else:
        instance_ids_tree = np.zeros(0, dtype=np.int32)

    # 6. Mapear a nube original
    instance_ids = np.zeros(len(data["xyz"]), dtype=np.int32)
    if n_tree_pts > 0:
        instance_ids[tree_mask] = instance_ids_tree

    # 7. Métricas
    binary_labels = get_binary_labels(data["classification"])
    valid = binary_labels != -1

    # Métricas semánticas
    sem_metrics = compute_semantic_metrics(
        binary_labels, pred_binary, mask_valid=valid,
    )

    # Métricas de instancia (solo sobre puntos válidos)
    inst_metrics = compute_instance_metrics_plot(
        instance_ids[valid],
        data["tree_id"][valid],
        iou_threshold=cfg.evaluation.iou_threshold,
    )

    n_pred_trees = inst_metrics["n_pred_trees"]
    n_gt_trees = inst_metrics["n_gt_trees"]
    logger.info(f"    Pred trees: {n_pred_trees}, GT trees: {n_gt_trees}, "
                f"F1: {inst_metrics['f1']:.3f}")

    # 8. Guardar predicciones
    pred_dir = output_dir / "predictions" / "rf"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        pred_dir / f"{plot_name}_instances.npz",
        instance_ids=instance_ids,
        semantic_pred=pred_binary,
    )

    # Combinar métricas
    result = {
        "plot": plot_name,
        "institution": institution,
        **{f"sem_{k}": v for k, v in sem_metrics.items()
           if k not in ("confusion_matrix",)},
        **inst_metrics,
    }
    return result


def run_rf_pipeline(cfg, split: str = "val", single_plot: str = None):
    """Ejecuta el pipeline RF + Watershed sobre los plots del split dado."""
    output_dir = Path(cfg.paths.output_dir)
    dataset_root = Path(cfg.paths.dataset_root)

    # Setup logging
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rf_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_dir / f"rf_pipeline_{split}.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    # Cargar modelo RF
    model_path = output_dir / "models" / "rf_model.joblib"
    if not model_path.exists():
        logger.error(f"RF model not found: {model_path}")
        logger.error("Run train_rf.py first.")
        return
    rf_model = joblib.load(model_path)
    logger.info(f"Loaded RF model: {model_path}")

    # Determinar plots
    if single_plot:
        plot_paths = [dataset_root / single_plot]
    else:
        dev_paths, test_paths = load_splits(dataset_root)
        if split == "val":
            _, plot_paths = get_train_val_split(
                dev_paths,
                val_fraction=cfg.data.val_split,
                random_state=cfg.data.random_state,
            )
        elif split == "test":
            plot_paths = test_paths
        else:
            plot_paths = dev_paths

    logger.info(f"=== RF Pipeline — {split} set — {len(plot_paths)} plots ===")

    all_metrics = []
    t_start = time.time()

    for las_path in tqdm(plot_paths, desc=f"RF pipeline ({split})"):
        try:
            result = _process_single_plot(
                las_path, rf_model, cfg, output_dir, logger,
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

    # Tabla de métricas
    df = pd.DataFrame(all_metrics)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"rf_metrics_{split}.csv"
    df.to_csv(csv_path, index=False)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Results — {split} set ({len(df)} plots, {t_total:.0f}s)")
    logger.info(f"{'=' * 70}")

    # Promedios
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
        logger.info(f"    F1 weight:  {df['sem_f1_weighted'].mean():.4f}")

    logger.info(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF + Watershed 3D pipeline")
    parser.add_argument("--split", default="val", choices=["val", "test", "dev"])
    parser.add_argument("--plot", default=None,
                        help="Single plot path relative to dataset root")
    args = parser.parse_args()

    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
    )
    run_rf_pipeline(cfg, split=args.split, single_plot=args.plot)
