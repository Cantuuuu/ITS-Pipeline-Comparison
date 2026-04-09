"""
Entrena el Random Forest para clasificación semántica binaria (árbol/no-árbol)
sobre los plots de entrenamiento de FOR-instance.

Flujo:
  1. Cargar splits (train/val)
  2. Para cada plot: load_las -> process_plot -> compute_features_for_plot
  3. Submuestreo balanceado (max MAX_POINTS_PER_CLASS por clase)
  4. Entrenar RF con hiperparámetros de config.yaml
  5. Evaluar en val
  6. Guardar modelo y feature importances

Hiperparámetros (Weinmann et al. 2017):
  - n_estimators=200: suficiente para convergencia con 28 features
  - max_depth=None: crecimiento completo, el ensemble regulariza
  - class_weight='balanced': compensa desbalance árbol/no-árbol
"""

import sys
import time
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from forest_its.data.dataset import load_las, get_binary_labels, load_splits
from forest_its.data.splits import get_train_val_split
from forest_its.preprocessing.normalize_height import process_plot
from forest_its.preprocessing.features_rf import (
    compute_features_for_plot, FEATURE_NAMES_28,
)
from forest_its.evaluation.semantic_metrics import compute_semantic_metrics


def _setup_logging(output_dir: Path) -> logging.Logger:
    """Configura logging a archivo y consola."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_rf")
    logger.setLevel(logging.INFO)

    # Limpiar handlers previos
    logger.handlers.clear()

    fh = logging.FileHandler(log_dir / "train_rf.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    return logger


def _load_and_extract(las_path: Path, cfg, output_dir: Path, logger):
    """Carga un plot, normaliza altura, extrae features y labels válidos."""
    data = load_las(las_path)
    data["_plot_stem"] = las_path.stem

    # Normalizar altura
    process_plot(
        data,
        resolution_dtm=cfg.preprocessing.dtm_resolution,
        smooth_window=cfg.preprocessing.smooth_window,
        hag_min=cfg.preprocessing.hag_min,
        hag_max=cfg.preprocessing.hag_max,
    )

    # Extraer features (con cache)
    features = compute_features_for_plot(data, cfg, output_dir)

    # Labels válidos
    binary_labels = get_binary_labels(data["classification"])
    valid_mask = binary_labels != -1
    labels = binary_labels[valid_mask]

    return features, labels


def train_rf(cfg):
    """Entrena el Random Forest sobre los plots de train."""
    output_dir = Path(cfg.paths.output_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logging(output_dir)
    t_start = time.time()

    dataset_root = Path(cfg.paths.dataset_root)
    dev_paths, _ = load_splits(dataset_root)
    train_paths, val_paths = get_train_val_split(
        dev_paths,
        val_fraction=cfg.data.val_split,
        random_state=cfg.data.random_state,
    )

    logger.info(f"Train plots: {len(train_paths)}, Val plots: {len(val_paths)}")

    # --- Extraer features de train ---
    logger.info("=== Extracting features (train) ===")
    all_features = []
    all_labels = []

    for las_path in tqdm(train_paths, desc="Train features"):
        try:
            feats, labels = _load_and_extract(las_path, cfg, output_dir, logger)
            all_features.append(feats)
            all_labels.append(labels)
        except Exception as e:
            logger.error(f"  ERROR {las_path.stem}: {e}")
            continue

    X_train = np.concatenate(all_features, axis=0)
    y_train = np.concatenate(all_labels, axis=0)

    n_tree = (y_train == 1).sum()
    n_notree = (y_train == 0).sum()
    logger.info(f"Train total: {len(y_train)} pts, "
                f"tree={n_tree}, non-tree={n_notree}")

    # --- Submuestreo balanceado ---
    max_per_class = cfg.rf.max_points_per_class
    rng = np.random.RandomState(cfg.rf.random_state)

    idx_tree = np.where(y_train == 1)[0]
    idx_notree = np.where(y_train == 0)[0]

    if len(idx_tree) > max_per_class:
        idx_tree = rng.choice(idx_tree, max_per_class, replace=False)
    if len(idx_notree) > max_per_class:
        idx_notree = rng.choice(idx_notree, max_per_class, replace=False)

    idx_sub = np.sort(np.concatenate([idx_tree, idx_notree]))
    X_train_sub = X_train[idx_sub]
    y_train_sub = y_train[idx_sub]

    logger.info(f"After subsampling: {len(y_train_sub)} pts "
                f"(tree={len(idx_tree)}, non-tree={len(idx_notree)})")

    # --- Entrenar RF ---
    logger.info("=== Training Random Forest ===")
    clf = RandomForestClassifier(
        n_estimators=cfg.rf.n_estimators,
        max_depth=cfg.rf.max_depth,
        class_weight=cfg.rf.class_weight,
        n_jobs=cfg.rf.n_jobs,
        random_state=cfg.rf.random_state,
        verbose=1,
    )

    t_train = time.time()
    clf.fit(X_train_sub, y_train_sub)
    t_train_end = time.time()
    logger.info(f"Training time: {t_train_end - t_train:.1f}s")

    # --- Guardar modelo ---
    model_path = model_dir / "rf_model.joblib"
    joblib.dump(clf, model_path)
    logger.info(f"Model saved: {model_path}")

    # --- Feature importances ---
    importances = clf.feature_importances_
    imp_df = pd.DataFrame({
        "feature": FEATURE_NAMES_28,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    imp_path = model_dir / "rf_feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)

    logger.info("\nTop-10 feature importances:")
    for _, row in imp_df.head(10).iterrows():
        logger.info(f"  {row['feature']:30s}  {row['importance']:.4f}")

    # --- Evaluar en validación ---
    if val_paths:
        logger.info(f"\n=== Evaluating on {len(val_paths)} val plots ===")
        all_val_feats = []
        all_val_labels = []

        for las_path in tqdm(val_paths, desc="Val features"):
            try:
                feats, labels = _load_and_extract(
                    las_path, cfg, output_dir, logger
                )
                all_val_feats.append(feats)
                all_val_labels.append(labels)
            except Exception as e:
                logger.error(f"  ERROR {las_path.stem}: {e}")
                continue

        X_val = np.concatenate(all_val_feats, axis=0)
        y_val = np.concatenate(all_val_labels, axis=0)

        y_pred_val = clf.predict(X_val)
        sem_metrics = compute_semantic_metrics(y_val, y_pred_val)

        logger.info(f"\nValidation metrics:")
        logger.info(f"  Accuracy:   {sem_metrics['accuracy']:.4f}")
        logger.info(f"  mIoU:       {sem_metrics['miou']:.4f}")
        logger.info(f"  IoU tree:   {sem_metrics['iou_tree']:.4f}")
        logger.info(f"  IoU notree: {sem_metrics['iou_notree']:.4f}")
        logger.info(f"  F1 weight:  {sem_metrics['f1_weighted']:.4f}")

    t_total = time.time() - t_start
    logger.info(f"\nTotal time: {t_total:.1f}s")

    return clf


if __name__ == "__main__":
    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
    )
    train_rf(cfg)
