"""
Script de exploración del dataset FOR-instance.

Verifica la lectura correcta del dataset, imprime estadísticas por
carpeta institucional y confirma la presencia de los campos requeridos
(Classification, treeID). Usar como primer paso de validación.

Modos:
    python -m forest_its.scripts.explore_dataset
        — exploración detallada de un plot por colección.
    python -m forest_its.scripts.explore_dataset --density-summary
        — recorre TODOS los plots y reporta densidad
          (mín / mediana / máx, n_plots) por colección y rango global.
"""

import sys
import argparse
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


def summarize_density_by_collection(dataset_root: Path, csv_out: Path = None):
    """
    Itera todos los plots del dataset FOR-instance, calcula densidad
    (puntos / área XY en m²) por plot y reporta mín / mediana / máx
    por colección más el rango global.

    La colección se infiere del nombre de la carpeta padre del .las
    (NIBIO, CULS, TUWIEN, RMIT, SCION).

    Si `csv_out` no es None, guarda además una tabla citable desde el
    paper con columnas `collection,n_plots,min,median,max` y una fila
    final `GLOBAL`.
    """
    print("=" * 70)
    print("FOR-instance — Density summary (all plots)")
    print("=" * 70)
    print(f"Dataset root: {dataset_root}")
    print()

    dev_paths, test_paths = load_splits(dataset_root)
    all_paths = list(dev_paths) + list(test_paths)
    print(f"Total plots: {len(all_paths)} "
          f"(dev={len(dev_paths)}, test={len(test_paths)})")
    print()

    per_collection = defaultdict(list)
    failed = []

    for las_path in all_paths:
        collection = las_path.parent.name
        try:
            data = load_las(las_path)
        except Exception as e:
            failed.append((las_path.name, str(e)))
            continue

        xyz = data["xyz"]
        n_points = len(xyz)
        if n_points == 0:
            failed.append((las_path.name, "empty point cloud"))
            continue

        x_span = float(xyz[:, 0].max() - xyz[:, 0].min())
        y_span = float(xyz[:, 1].max() - xyz[:, 1].min())
        area = x_span * y_span
        if area <= 0:
            failed.append((las_path.name, "zero-area bbox"))
            continue

        density = n_points / area
        per_collection[collection].append(density)

    # Resumen por colección
    print(f"{'Collection':<12} {'n_plots':>8} {'min':>12} {'median':>12} "
          f"{'max':>12}")
    print("-" * 60)

    rows = []
    all_densities = []
    for collection in sorted(per_collection.keys()):
        densities = np.array(per_collection[collection])
        all_densities.extend(densities.tolist())
        row = {
            "collection": collection,
            "n_plots": int(len(densities)),
            "min": float(densities.min()),
            "median": float(np.median(densities)),
            "max": float(densities.max()),
        }
        rows.append(row)
        print(
            f"{collection:<12} {row['n_plots']:>8d} "
            f"{row['min']:>12.1f} {row['median']:>12.1f} "
            f"{row['max']:>12.1f}"
        )

    print("-" * 60)
    if all_densities:
        all_densities = np.array(all_densities)
        global_row = {
            "collection": "GLOBAL",
            "n_plots": int(len(all_densities)),
            "min": float(all_densities.min()),
            "median": float(np.median(all_densities)),
            "max": float(all_densities.max()),
        }
        rows.append(global_row)
        print(
            f"{'GLOBAL':<12} {global_row['n_plots']:>8d} "
            f"{global_row['min']:>12.1f} {global_row['median']:>12.1f} "
            f"{global_row['max']:>12.1f}"
        )
        print()
        print(f"Global range (min–max): "
              f"{global_row['min']:.0f}–{global_row['max']:.0f} pts/m²")
    else:
        print("No valid plots processed.")

    if csv_out is not None and rows:
        csv_out = Path(csv_out)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_out, index=False)
        print(f"\nCSV saved: {csv_out}")

    if failed:
        print()
        print(f"Failed to process {len(failed)} plots:")
        for name, msg in failed:
            print(f"  {name}: {msg}")

    print("=" * 70)


def _resolve_dataset_root(cli_root: str = None) -> Path:
    if cli_root:
        return Path(cli_root)
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(
            Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        )
        return Path(cfg.paths.dataset_root)
    except Exception:
        # Fallback: repo-local FORinstance_dataset/ si existe
        local = Path(__file__).resolve().parent.parent.parent / "FORinstance_dataset"
        if local.exists():
            return local
        return Path("C:/Users/cantu/Downloads/FORinstance_dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOR-instance dataset explorer")
    parser.add_argument(
        "--density-summary", action="store_true",
        help="Recorrer todos los plots y reportar densidad por colección.",
    )
    parser.add_argument(
        "--dataset-root", default=None,
        help="Override del dataset root (por defecto: config.yaml o ./FORinstance_dataset).",
    )
    parser.add_argument(
        "--density-csv", default=None,
        help="Ruta de salida para exportar el resumen de densidades a CSV.",
    )
    args = parser.parse_args()

    dataset_root = _resolve_dataset_root(args.dataset_root)

    if args.density_summary:
        csv_out = Path(args.density_csv) if args.density_csv else None
        summarize_density_by_collection(dataset_root, csv_out=csv_out)
    else:
        explore_dataset(dataset_root)
