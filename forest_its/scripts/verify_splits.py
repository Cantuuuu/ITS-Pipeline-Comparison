"""
Auditoría de los splits de FOR-instance V1 contra los conteos oficiales
reportados en la Tabla 1 del paper.

Tras invocar `load_splits`, recorre el split test y cuenta:
  - número de plots por colección
  - número de árboles únicos (treeID > 0) por colección, restringido a
    puntos con binary_label != -1 (excluye clases 0 y 3 siguiendo el
    protocolo oficial del paper).

Los valores esperados provienen de la Tabla 1 del paper (§3.1):
    NIBIO:  6 plots / 161 árboles
    CULS:   1 plot  /  20 árboles
    TUWIEN: 1 plot  /  35 árboles
    RMIT:   1 plot  /  64 árboles
    SCION:  2 plots /  43 árboles
    TOTAL: 11 plots / 323 árboles

Uso:
    python -m forest_its.scripts.verify_splits
    python -m forest_its.scripts.verify_splits --dataset-root /path/to/FORinstance_dataset

Salida: tabla por consola con diffs respecto a los valores esperados y
código de salida != 0 si alguna colección no coincide.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from forest_its.data.dataset import load_las, get_binary_labels, load_splits


EXPECTED_TEST = {
    "NIBIO":  {"plots": 6, "trees": 161},
    "CULS":   {"plots": 1, "trees": 20},
    "TUWIEN": {"plots": 1, "trees": 35},
    "RMIT":   {"plots": 1, "trees": 64},
    "SCION":  {"plots": 2, "trees": 43},
}
EXPECTED_TOTAL = {"plots": 11, "trees": 323}


def count_trees_per_collection(paths):
    """
    Cuenta plots y árboles únicos (treeID > 0, sobre puntos válidos) por
    colección. La colección se infiere del nombre de la carpeta padre.
    """
    per_collection = defaultdict(lambda: {"plots": 0, "trees": 0})
    for path in paths:
        collection = path.parent.name
        try:
            data = load_las(path)
        except Exception as e:
            print(f"  [SKIP] {path.name}: {e}")
            continue

        binary_labels = get_binary_labels(data["classification"])
        valid_mask = binary_labels != -1
        tree_ids = data["tree_id"][valid_mask]
        unique_trees = np.unique(tree_ids[tree_ids > 0])

        per_collection[collection]["plots"] += 1
        per_collection[collection]["trees"] += int(len(unique_trees))

    return dict(per_collection)


def _format_diff(actual: int, expected: int) -> str:
    if actual == expected:
        return "OK"
    return f"MISMATCH (+{actual - expected})" if actual > expected else f"MISMATCH ({actual - expected})"


def verify_test_split(dataset_root: Path) -> bool:
    dev_paths, test_paths = load_splits(dataset_root)
    print(f"Splits cargados: dev={len(dev_paths)}, test={len(test_paths)}")
    print()

    counts = count_trees_per_collection(test_paths)

    header = f"{'Collection':<10} {'plots':>6} {'exp':>5} {'trees':>7} {'exp':>5}   status"
    print("TEST split — counts por colección")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    all_ok = True
    total_plots = 0
    total_trees = 0
    for collection, expected in EXPECTED_TEST.items():
        actual = counts.get(collection, {"plots": 0, "trees": 0})
        plot_status = _format_diff(actual["plots"], expected["plots"])
        tree_status = _format_diff(actual["trees"], expected["trees"])
        status = "OK" if plot_status == "OK" and tree_status == "OK" else (
            f"plots:{plot_status} trees:{tree_status}"
        )
        if status != "OK":
            all_ok = False
        print(
            f"{collection:<10} {actual['plots']:>6} {expected['plots']:>5} "
            f"{actual['trees']:>7} {expected['trees']:>5}   {status}"
        )
        total_plots += actual["plots"]
        total_trees += actual["trees"]

    # Colecciones inesperadas (p. ej. NIBIO2 si aparece por accidente)
    extras = set(counts.keys()) - set(EXPECTED_TEST.keys())
    for extra in sorted(extras):
        actual = counts[extra]
        print(
            f"{extra:<10} {actual['plots']:>6} {'-':>5} "
            f"{actual['trees']:>7} {'-':>5}   UNEXPECTED"
        )
        all_ok = False
        total_plots += actual["plots"]
        total_trees += actual["trees"]

    print("-" * len(header))
    total_status = "OK" if (
        total_plots == EXPECTED_TOTAL["plots"]
        and total_trees == EXPECTED_TOTAL["trees"]
    ) else "MISMATCH"
    if total_status != "OK":
        all_ok = False
    print(
        f"{'TOTAL':<10} {total_plots:>6} {EXPECTED_TOTAL['plots']:>5} "
        f"{total_trees:>7} {EXPECTED_TOTAL['trees']:>5}   {total_status}"
    )
    print()
    return all_ok


def _resolve_dataset_root(cli_root: str = None) -> Path:
    if cli_root:
        return Path(cli_root)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    )
    return Path(cfg.paths.dataset_root)


def main():
    parser = argparse.ArgumentParser(
        description="Auditoría de los splits de FOR-instance contra Tabla 1"
    )
    parser.add_argument(
        "--dataset-root", default=None,
        help="Override del dataset root (por defecto: config.yaml).",
    )
    args = parser.parse_args()

    dataset_root = _resolve_dataset_root(args.dataset_root)
    print(f"Dataset root: {dataset_root}")
    print()

    ok = verify_test_split(dataset_root)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
