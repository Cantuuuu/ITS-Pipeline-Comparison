"""
Grid search independiente por flujo sobre los parametros del Watershed 3D.

Motivacion:
  Los tres flujos (baseline, rf, pointnet2) comparten el mismo
  segmentador (Watershed 3D), pero reciben puntos de entrada distintos:
    - Flujo A (baseline):  todos los puntos validos
    - Flujo B (rf):        solo puntos clasificados como arbol por RF
    - Flujo C (pointnet2): solo puntos clasificados como arbol por PointNet++

  Como la distribucion de puntos de entrada difiere, los parametros
  optimos del watershed pueden diferir tambien. Este modulo calibra
  esos parametros por flujo mediante grid search exhaustivo sobre el
  conjunto val, maximizando el F1 de segmentacion de instancias
  (IoU 3D >= 0.5).

Uso:
  # Requiere que los pipelines hayan corrido previamente sobre val
  # para que existan las predicciones semanticas en
  # output/predictions/{rf,pointnet2}/
  python -m forest_its.evaluation.grid_search --methods baseline rf pointnet2

  # Solo un metodo
  python -m forest_its.evaluation.grid_search --methods rf

Espacio de busqueda (configurable en GRID):
  voxel_size:         [0.1, 0.2, 0.3]  m
  gaussian_sigma:     [0.3, 0.5, 1.0]
  min_crown_radius_m: [0.5, 1.0, 1.5, 2.0]  m

Total: 36 combinaciones por flujo.

Salida:
  - output/results/grid_search_{method}.csv  (todas las combos + F1)
  - output/results/grid_search_best_params.csv  (resumen)
"""

import sys
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from forest_its.data.dataset import load_las, get_binary_labels, load_splits
from forest_its.data.splits import get_train_val_split
from forest_its.preprocessing.normalize_height import compute_hag
from forest_its.segmentation.watershed3d import watershed3d
from forest_its.segmentation.watershed3d_density import watershed3d_density
from forest_its.evaluation.instance_metrics import compute_instance_metrics_plot


# Grid de busqueda. Modificar aqui para ampliar/reducir.
GRID = {
    "voxel_size": [0.1, 0.2, 0.3],
    "gaussian_sigma": [0.3, 0.5, 1.0],
    "min_crown_radius_m": [0.5, 1.0, 1.5, 2.0],
}

# Grid extendido para rf_density (incluye top_band_m)
GRID_DENSITY = {
    "voxel_size": [0.1, 0.2, 0.3],
    "gaussian_sigma": [0.3, 0.5, 1.0],
    "min_crown_radius_m": [0.5, 1.0, 1.5, 2.0],
    "top_band_m": [2.0, 3.0, 5.0],
}


def _prepare_plot(method: str, las_path: Path, cfg, output_dir: Path) -> dict:
    """
    Prepara los tensores necesarios para correr watershed sobre un plot
    segun el metodo elegido. Computa HAG una sola vez y carga la prediccion
    semantica pre-calculada si el metodo lo requiere.

    Returns:
        dict con:
          - n_total: numero de puntos de la nube completa
          - valid_mask: (N,) bool, puntos con label != -1
          - gt_tree_id: (N,) int, treeID GT
          - points_for_ws: (M, 3) float, (X, Y, HAG) que entran al watershed
          - ws_mask: (N,) bool, mascara de puntos que entraron al watershed
            (para mapear instance_ids de vuelta a la nube completa)
    """
    data = load_las(las_path)
    binary_labels = get_binary_labels(data["classification"])
    valid_mask = binary_labels != -1
    n_total = len(data["xyz"])

    # HAG sobre la nube completa (necesario para filtro min_hag y para pasar
    # a watershed). Calcular solo una vez por plot.
    hag_full = compute_hag(
        data["xyz"],
        classification=data["classification"],
        resolution=cfg.preprocessing.dtm_resolution,
        smooth_window=cfg.preprocessing.smooth_window,
        hag_min=cfg.preprocessing.hag_min,
        hag_max=cfg.preprocessing.hag_max,
    )

    if method == "baseline":
        # Todos los puntos validos entran al watershed
        ws_mask = valid_mask.copy()
    else:
        # Cargar prediccion semantica pre-calculada
        # *_density reutiliza las predicciones del método base
        pred_method = {"rf_density": "rf", "pointnet2_density": "pointnet2"}.get(method, method)
        plot_name = f"{las_path.parent.name}__{las_path.stem}"
        pred_file = output_dir / "predictions" / pred_method / f"{plot_name}_instances.npz"
        if not pred_file.exists():
            raise FileNotFoundError(
                f"Prediccion semantica no encontrada para {plot_name} ({method}).\n"
                f"  Esperado: {pred_file}\n"
                f"  Corre el pipeline primero:\n"
                f"    python -m forest_its.methods.{method}.run_{method}_pipeline --split val"
            )
        pred = np.load(pred_file)
        semantic_pred = pred["semantic_pred"]  # (N,) {-1, 0, 1}
        # Filtro idéntico al de los pipelines: sem==1 & HAG>=min_hag
        ws_mask = (
            (semantic_pred == 1)
            & (hag_full >= cfg.preprocessing.min_hag_tree_filter)
            & valid_mask
        )

    xyz = data["xyz"]
    points_for_ws = np.column_stack([
        xyz[ws_mask, 0],
        xyz[ws_mask, 1],
        hag_full[ws_mask],
    ]).astype(np.float64)

    return {
        "plot": f"{las_path.parent.name}__{las_path.stem}",
        "n_total": n_total,
        "valid_mask": valid_mask,
        "gt_tree_id": data["tree_id"],
        "points_for_ws": points_for_ws,
        "ws_mask": ws_mask,
    }


def _evaluate_params(
    plot_info: dict,
    voxel_size: float,
    gaussian_sigma: float,
    min_crown_radius_m: float,
    min_tree_height: float,
    min_points_per_tree: int,
    iou_threshold: float,
    use_density: bool = False,
    top_band_m: float = 3.0,
) -> float:
    """
    Corre watershed con los parametros dados y retorna F1 de instancia
    sobre un plot.
    """
    points_for_ws = plot_info["points_for_ws"]
    if len(points_for_ws) == 0:
        return 0.0

    if use_density:
        inst_ids_ws = watershed3d_density(
            points_for_ws,
            voxel_size=voxel_size,
            gaussian_sigma=gaussian_sigma,
            min_crown_radius_m=min_crown_radius_m,
            min_tree_height=min_tree_height,
            min_points_per_tree=min_points_per_tree,
            top_band_m=top_band_m,
        )
    else:
        inst_ids_ws = watershed3d(
            points_for_ws,
            voxel_size=voxel_size,
            gaussian_sigma=gaussian_sigma,
            min_crown_radius_m=min_crown_radius_m,
            min_tree_height=min_tree_height,
            min_points_per_tree=min_points_per_tree,
        )

    # Mapear a la nube completa
    instance_ids = np.zeros(plot_info["n_total"], dtype=np.int32)
    instance_ids[plot_info["ws_mask"]] = inst_ids_ws

    valid = plot_info["valid_mask"]
    metrics = compute_instance_metrics_plot(
        instance_ids[valid],
        plot_info["gt_tree_id"][valid],
        iou_threshold=iou_threshold,
    )
    return float(metrics["f1"])


def run_grid_search(method: str, cfg, output_dir: Path, dataset_root: Path) -> dict:
    """
    Grid search sobre watershed params para un flujo. Retorna los mejores
    parametros (los que maximizan el F1 medio de instancia sobre val plots).
    """
    use_density = method in ("rf_density", "pointnet2_density")

    # Val set = fraccion fija del split dev (igual que en los pipelines)
    dev_paths, _ = load_splits(dataset_root)
    _, val_paths = get_train_val_split(
        dev_paths,
        val_fraction=cfg.data.val_split,
        random_state=cfg.data.random_state,
    )

    print(f"\n{'=' * 70}")
    print(f"Grid search: {method} — {len(val_paths)} val plots")
    print(f"{'=' * 70}")

    # Pre-cargar plots (HAG y mascaras) para no recalcular por combinacion
    print("Preparando plots val (cargando LAS, computando HAG)...")
    plot_data = []
    for p in tqdm(val_paths, desc="  Preparing"):
        try:
            info = _prepare_plot(method, p, cfg, output_dir)
            plot_data.append(info)
        except Exception as e:
            print(f"  [SKIP] {p.stem}: {e}")
            continue

    if not plot_data:
        print(f"  [ERROR] No hay plots val disponibles para {method}.")
        return None

    if use_density:
        grid = GRID_DENSITY
        param_combos = list(itertools.product(
            grid["voxel_size"],
            grid["gaussian_sigma"],
            grid["min_crown_radius_m"],
            grid["top_band_m"],
        ))
        param_keys = ["voxel_size", "gaussian_sigma", "min_crown_radius_m", "top_band_m"]
    else:
        grid = GRID
        param_combos = list(itertools.product(
            grid["voxel_size"],
            grid["gaussian_sigma"],
            grid["min_crown_radius_m"],
        ))
        param_keys = ["voxel_size", "gaussian_sigma", "min_crown_radius_m"]

    n_runs = len(param_combos) * len(plot_data)
    print(f"Evaluando {len(param_combos)} combinaciones "
          f"x {len(plot_data)} plots = {n_runs} runs (paralelo)")

    # Aplanar (combo, plot) — paralelizar con joblib/loky.
    # Cada tarea es un watershed3d + métricas: CPU-bound puro y GIL-libre.
    tasks = [
        (combo_idx, plot_idx, combo, info)
        for combo_idx, combo in enumerate(param_combos)
        for plot_idx, info in enumerate(plot_data)
    ]

    def _task(combo_idx, plot_idx, combo, info):
        params = dict(zip(param_keys, combo))
        f1 = _evaluate_params(
            info,
            voxel_size=params["voxel_size"],
            gaussian_sigma=params["gaussian_sigma"],
            min_crown_radius_m=params["min_crown_radius_m"],
            min_tree_height=cfg.watershed.min_tree_height,
            min_points_per_tree=cfg.watershed.min_points_per_tree,
            iou_threshold=cfg.evaluation.iou_threshold,
            use_density=use_density,
            top_band_m=params.get("top_band_m", 3.0),
        )
        return combo_idx, plot_idx, f1

    flat = Parallel(n_jobs=-1, backend="loky")(
        delayed(_task)(*t) for t in tqdm(tasks, desc=f"  Grid {method}")
    )

    # Reagrupar por combo
    f1_matrix = np.zeros((len(param_combos), len(plot_data)), dtype=np.float64)
    for combo_idx, plot_idx, f1 in flat:
        f1_matrix[combo_idx, plot_idx] = f1

    results = []
    best_f1 = -1.0
    best_params = None
    for combo_idx, combo in enumerate(param_combos):
        params = dict(zip(param_keys, combo))
        mean_f1 = float(np.mean(f1_matrix[combo_idx]))
        std_f1 = float(np.std(f1_matrix[combo_idx]))
        results.append({
            **params,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "n_plots": len(plot_data),
        })
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = params.copy()

    # Guardar tabla completa
    df = pd.DataFrame(results).sort_values("mean_f1", ascending=False)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"grid_search_{method}.csv"
    df.to_csv(out_path, index=False)

    print(f"\n  Best params for {method}: F1 = {best_f1:.4f}")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"  Saved: {out_path}")

    return {**best_params, "mean_f1": best_f1}


def main():
    parser = argparse.ArgumentParser(
        description="Grid search de watershed params por flujo (sobre val set)"
    )
    parser.add_argument(
        "--methods", nargs="+", default=["baseline", "rf", "pointnet2"],
        choices=["baseline", "rf", "pointnet2", "rf_density", "pointnet2_density"],
        help="Flujos a calibrar",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path a config.yaml (default: forest_its/configs/config.yaml)",
    )
    args = parser.parse_args()

    if args.config:
        cfg = OmegaConf.load(args.config)
    else:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        )

    output_dir = Path(cfg.paths.output_dir)
    dataset_root = Path(cfg.paths.dataset_root)

    best_per_method = {}
    for method in args.methods:
        try:
            best = run_grid_search(method, cfg, output_dir, dataset_root)
            if best:
                best_per_method[method] = best
        except Exception as e:
            print(f"\n[FAILED] {method}: {e}")
            import traceback
            traceback.print_exc()

    if best_per_method:
        summary_rows = [
            {"method": method, **params}
            for method, params in best_per_method.items()
        ]
        summary_df = pd.DataFrame(summary_rows)
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / "grid_search_best_params.csv"
        summary_df.to_csv(out, index=False)
        print(f"\n{'=' * 70}")
        print(f"Grid search summary")
        print(f"{'=' * 70}")
        print(summary_df.to_string(index=False))
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
