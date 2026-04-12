"""
Utilidad para cargar los parametros optimos del watershed por flujo,
producidos por `grid_search.py`.

Los pipelines baseline/rf/pointnet2 consultan esta utilidad antes de
ejecutar el watershed en su etapa de instancias. Si no existe el CSV
de mejores parametros (o no contiene una fila para el metodo), se
lanza un error explicito pidiendo correr `grid_search.py` primero.

Esto fuerza el flujo de trabajo correcto del paper:

  1. train_rf.py / train_pointnet2.py
  2. run_rf_pipeline.py / run_pointnet2_pipeline.py --stage semantic
  3. grid_search.py  (calibra watershed por flujo sobre val set)
  4. run_*_pipeline.py --stage instance  (usa best_params.csv)
"""

from pathlib import Path
import pandas as pd


class MissingBestParamsError(RuntimeError):
    """Se lanza cuando no existen parametros calibrados para un flujo."""


def load_best_watershed_params(method: str, cfg) -> dict:
    """
    Retorna los parametros de watershed calibrados para un flujo.

    Lee `output/results/grid_search_best_params.csv`, que es generado por
    `forest_its.evaluation.grid_search`. Si el archivo no existe o no
    contiene una fila para el metodo solicitado, lanza MissingBestParamsError.

    Args:
        method: "baseline", "rf", "pointnet2".
        cfg:    OmegaConf config.

    Returns:
        Dict con las claves: voxel_size, gaussian_sigma, min_crown_radius_m,
        min_tree_height, min_points_per_tree. Los tres primeros provienen del
        grid search; los dos ultimos se toman del config (no forman parte de
        la busqueda para mantener la dimensionalidad manejable).
    """
    best_file = Path(cfg.paths.output_dir) / "results" / "grid_search_best_params.csv"
    if not best_file.exists():
        raise MissingBestParamsError(
            f"No se encuentra {best_file}.\n"
            f"Corre primero el grid search para calibrar el watershed:\n"
            f"  python -m forest_its.evaluation.grid_search --methods {method}"
        )

    df = pd.read_csv(best_file)
    row = df[df["method"] == method]
    if row.empty:
        raise MissingBestParamsError(
            f"{best_file} existe pero no contiene fila para method='{method}'.\n"
            f"Corre el grid search para ese flujo:\n"
            f"  python -m forest_its.evaluation.grid_search --methods {method}"
        )

    row = row.iloc[0]
    return {
        "voxel_size": float(row["voxel_size"]),
        "gaussian_sigma": float(row["gaussian_sigma"]),
        "min_crown_radius_m": float(row["min_crown_radius_m"]),
        "min_tree_height": float(cfg.watershed.min_tree_height),
        "min_points_per_tree": int(cfg.watershed.min_points_per_tree),
    }
