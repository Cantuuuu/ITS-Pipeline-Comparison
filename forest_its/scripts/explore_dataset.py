"""
Script de exploración del dataset FOR-instance.

Verifica la lectura correcta del dataset, imprime estadísticas por
carpeta institucional y confirma la presencia de los campos requeridos
(Classification, treeID). Usar como primer paso de validación.

Uso:
    python scripts/explore_dataset.py
"""

import sys
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from collections import defaultdict

from forest_its.data.dataset import load_las, get_binary_labels, load_splits


def explore_dataset(dataset_root: Path):
    """Explora el dataset FOR-instance e imprime estadísticas."""

    print("=" * 70)
    print("FOR-instance Dataset Explorer")
    print("=" * 70)
    print(f"Dataset root: {dataset_root}")
    print()

    # --- a) Leer splits ---
    dev_paths, test_paths = load_splits(dataset_root)
    print(f"Split info (from data_split_metadata.csv):")
    print(f"  Dev plots:  {len(dev_paths)}")
    print(f"  Test plots: {len(test_paths)}")
    print(f"  Total:      {len(dev_paths) + len(test_paths)}")
    print()

    # Agrupar por carpeta institucional
    folders = defaultdict(list)
    for p in dev_paths + test_paths:
        folder = p.parent.name
        folders[folder].append(p)

    print(f"Folders: {sorted(folders.keys())}")
    print()

    # --- b) Explorar un plot por carpeta ---
    CLASS_NAMES = {
        0: "Unclassified",
        1: "Low-vegetation",
        2: "Terrain",
        3: "Out-points",
        4: "Stem",
        5: "Live-branches",
        6: "Woody-branches",
    }

    BINARY_NAMES = {-1: "excluir", 0: "no-arbol", 1: "arbol"}

    for folder_name in sorted(folders.keys()):
        folder_plots = folders[folder_name]
        las_path = folder_plots[0]  # primer .las disponible

        print("-" * 70)
        print(f"Folder: {folder_name} ({len(folder_plots)} plots)")
        print(f"  Sample: {las_path.name}")
        print()

        try:
            data = load_las(las_path)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            print()
            continue

        xyz = data["xyz"]
        classification = data["classification"]
        tree_id = data["tree_id"]
        intensity = data["intensity"]

        n_points = len(xyz)
        print(f"  Total points: {n_points:,}")

        # Distribución de clases
        print(f"  Classification distribution:")
        unique_classes, counts = np.unique(classification, return_counts=True)
        for cls, cnt in zip(unique_classes, counts):
            name = CLASS_NAMES.get(cls, f"Unknown({cls})")
            pct = 100.0 * cnt / n_points
            print(f"    {cls} ({name}): {cnt:>10,} ({pct:5.1f}%)")

        # Árboles únicos
        unique_trees = np.unique(tree_id)
        unique_trees_nonzero = unique_trees[unique_trees > 0]
        print(f"  Unique trees (treeID > 0): {len(unique_trees_nonzero)}")
        if 0 in unique_trees:
            n_zero = (tree_id == 0).sum()
            print(f"  Points with treeID=0 (no tree): {n_zero:,}")

        # Rango XYZ
        print(f"  XYZ range:")
        for i, axis in enumerate(["X", "Y", "Z"]):
            print(f"    {axis}: [{xyz[:, i].min():.2f}, {xyz[:, i].max():.2f}]"
                  f"  span={xyz[:, i].max() - xyz[:, i].min():.2f}")

        # Densidad
        x_span = xyz[:, 0].max() - xyz[:, 0].min()
        y_span = xyz[:, 1].max() - xyz[:, 1].min()
        area = x_span * y_span
        density = n_points / area if area > 0 else 0
        print(f"  Area XY: {area:.1f} m²")
        print(f"  Density: {density:.0f} pts/m²")

        # Intensidad
        print(f"  Intensity: [{intensity.min():.4f}, {intensity.max():.4f}] "
              f"(mean={intensity.mean():.4f})")

        # Distribución binaria
        binary_labels = get_binary_labels(classification)
        print(f"  Binary label distribution:")
        for label in [-1, 0, 1]:
            cnt = (binary_labels == label).sum()
            pct = 100.0 * cnt / n_points
            name = BINARY_NAMES[label]
            print(f"    {label:>2} ({name:>9}): {cnt:>10,} ({pct:5.1f}%)")

        # --- c) Advertencias ---
        if n_points == 0:
            print("  WARNING: Empty point cloud!")
        if classification.max() == 0 and classification.min() == 0:
            print("  WARNING: All Classification values are 0!")
        if tree_id.max() == 0:
            print("  WARNING: All treeID values are 0 — no annotated trees!")

        print()

    print("=" * 70)
    print("Exploration complete.")


if __name__ == "__main__":
    # Intentar cargar config, si no existe usar path por defecto
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(
            Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        )
        dataset_root = Path(cfg.paths.dataset_root)
    except Exception:
        dataset_root = Path("C:/Users/cantu/Downloads/FORinstance_dataset")

    explore_dataset(dataset_root)
