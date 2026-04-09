"""
Visualización de resultados de segmentación.

Genera visualizaciones 2D (vista cenital) de las instancias predichas
vs ground truth para inspección visual rápida.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from forest_its.data.dataset import load_las, get_binary_labels


def generate_random_cmap(n_colors: int, seed: int = 42) -> ListedColormap:
    """Genera un colormap aleatorio para instancias."""
    rng = np.random.RandomState(seed)
    colors = rng.rand(n_colors, 3)
    colors[0] = [0.8, 0.8, 0.8]  # ID 0 = gris (sin asignación)
    return ListedColormap(colors)


def plot_instances_2d(
    xyz: np.ndarray,
    instance_ids: np.ndarray,
    title: str = "Instance Segmentation",
    ax=None,
    point_size: float = 0.1,
):
    """
    Visualización cenital (XY) coloreada por instancia.

    Args:
        xyz: (N, 3) coordenadas.
        instance_ids: (N,) IDs de instancia.
        title: Título del plot.
        ax: Matplotlib axes (si None, crea nuevo).
        point_size: Tamaño de los puntos.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    n_instances = instance_ids.max() + 1
    cmap = generate_random_cmap(max(n_instances, 2))

    scatter = ax.scatter(
        xyz[:, 0], xyz[:, 1],
        c=instance_ids, cmap=cmap,
        s=point_size, marker=".", edgecolors="none",
    )
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    return ax


def visualize_comparison(
    las_path: Path,
    result_paths: dict,
    output_path: Path = None,
):
    """
    Compara GT vs predicciones de los 3 métodos para un plot.

    Args:
        las_path: Ruta al .las original.
        result_paths: Dict {method_name: path_to_result.npz}.
        output_path: Si se provee, guarda la figura.
    """
    data = load_las(las_path)
    binary_labels = get_binary_labels(data["classification"])
    valid = binary_labels != -1

    n_methods = len(result_paths) + 1  # +1 para GT
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))

    if n_methods == 1:
        axes = [axes]

    # Ground truth
    plot_instances_2d(
        data["xyz"][valid], data["tree_id"][valid],
        title=f"Ground Truth\n{las_path.stem}",
        ax=axes[0],
    )

    # Predicciones
    for i, (method_name, result_path) in enumerate(result_paths.items(), 1):
        if Path(result_path).exists():
            result = np.load(result_path)
            plot_instances_2d(
                data["xyz"][valid], result["instance_ids"][valid],
                title=f"{method_name}",
                ax=axes[i],
            )
        else:
            axes[i].set_title(f"{method_name}\n(no results)")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    )
    dataset_root = Path(cfg.paths.dataset_root)
    output_dir = Path(cfg.paths.output_dir)

    from forest_its.data.dataset import load_splits
    _, test_paths = load_splits(dataset_root)

    if test_paths:
        las_path = test_paths[0]
        plot_name = las_path.stem

        result_paths = {}
        for method in ["baseline", "rf", "pointnet2"]:
            rp = output_dir / method / f"{plot_name}_result.npz"
            if rp.exists():
                result_paths[method] = rp

        if result_paths:
            visualize_comparison(las_path, result_paths)
        else:
            print("No results found. Run the pipelines first.")
    else:
        print("No test plots found.")
