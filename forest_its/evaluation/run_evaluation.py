"""
Genera tabla de métricas comparativa entre los métodos evaluados.

Lee los archivos output_dir/results/{method}_metrics_{split}.csv
y genera una tabla comparativa por institución y promedio global.

Uso:
  python -m forest_its.evaluation.run_evaluation --methods baseline rf
  python -m forest_its.evaluation.run_evaluation --methods baseline rf pointnet2
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from forest_its.data.dataset import load_splits


def run_evaluation(cfg, methods: list, split: str = "val"):
    """
    Genera tabla comparativa leyendo CSVs de métricas previamente guardados.

    Args:
        cfg: Configuración.
        methods: Lista de nombres de métodos a comparar.
        split: 'val' o 'test'.
    """
    output_dir = Path(cfg.paths.output_dir)
    results_dir = output_dir / "results"

    all_dfs = []
    for method in methods:
        csv_path = results_dir / f"{method}_metrics_{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["method"] = method
            all_dfs.append(df)
            print(f"  Loaded: {csv_path} ({len(df)} plots)")
        else:
            print(f"  [SKIP] {csv_path} not found")

    if not all_dfs:
        print("No results found. Run the pipelines first.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # --- Tabla por institución ---
    print("\n" + "=" * 80)
    print(f"COMPARISON TABLE — {split} set")
    print("=" * 80)

    # Extraer institución del nombre del plot
    if "institution" not in combined.columns:
        combined["institution"] = combined["plot"].apply(
            lambda x: str(x).split("/")[0] if "/" in str(x) else "unknown"
        )

    agg_cols = {
        "precision": "mean",
        "recall": "mean",
        "f1": "mean",
        "n_gt_trees": "sum",
        "n_pred_trees": "sum",
        "coverage": "mean",
        "mean_iou_matched": "mean",
        "over_seg": "mean",
        "under_seg": "mean",
    }
    # Solo incluir columnas que existen en el dataframe
    agg_cols = {k: v for k, v in agg_cols.items() if k in combined.columns}

    inst_metrics = combined.groupby(["method", "institution"]).agg(agg_cols).round(4)

    print("\nPer-institution metrics:")
    print(inst_metrics.to_string())

    # --- Promedio global ponderado por número de árboles GT ---
    print("\n" + "-" * 80)
    print("Global averages (weighted by n_gt_trees):")
    print("-" * 80)

    for method in methods:
        mdf = combined[combined["method"] == method]
        if mdf.empty:
            continue
        total_gt = mdf["n_gt_trees"].sum()
        total_pred = mdf["n_pred_trees"].sum() if "n_pred_trees" in mdf.columns else 0
        if total_gt > 0:
            w_prec = (mdf["precision"] * mdf["n_gt_trees"]).sum() / total_gt
            w_rec = (mdf["recall"] * mdf["n_gt_trees"]).sum() / total_gt
            w_f1 = (mdf["f1"] * mdf["n_gt_trees"]).sum() / total_gt
            w_cov = (mdf["coverage"] * mdf["n_gt_trees"]).sum() / total_gt
        else:
            w_prec = w_rec = w_f1 = w_cov = 0.0

        mean_iou = mdf["mean_iou_matched"].mean() if "mean_iou_matched" in mdf.columns else float("nan")
        over_seg = mdf["over_seg"].mean() if "over_seg" in mdf.columns else float("nan")
        under_seg = mdf["under_seg"].mean() if "under_seg" in mdf.columns else float("nan")

        sem_str = ""
        if "sem_miou" in mdf.columns:
            sem_miou = mdf["sem_miou"].mean()
            sem_str = f"  mIoU_sem: {sem_miou:.4f}"

        print(f"  {method:12s} | P: {w_prec:.4f} | R: {w_rec:.4f} | "
              f"F1: {w_f1:.4f} | Cov: {w_cov:.4f} | "
              f"GT: {total_gt} | Pred: {total_pred} | "
              f"IoU_match: {mean_iou:.4f} | "
              f"OverSeg: {over_seg:.4f} | UnderSeg: {under_seg:.4f}"
              f"{sem_str}")

    # Guardar tabla combinada
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"comparison_table_{split}.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Generar tabla LaTeX
    _write_latex_table(combined, methods, split, results_dir)


def _write_latex_table(df, methods, split, results_dir):
    """Genera tabla LaTeX lista para el paper."""
    tex_path = results_dir / f"comparison_table_{split}.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Instance segmentation results on {split} set}}\n")
        f.write("\\begin{tabular}{l c c c c c c c c}\n")
        f.write("\\hline\n")
        f.write("Method & GT & Pred & Prec & Rec & F1 & Cov "
                "& IoU$_{\\text{match}}$ & OverSeg & UnderSeg \\\\\n")
        f.write("\\hline\n")

        for method in methods:
            mdf = df[df["method"] == method]
            if mdf.empty:
                continue
            total_gt = int(mdf["n_gt_trees"].sum())
            total_pred = int(mdf["n_pred_trees"].sum()) if "n_pred_trees" in mdf.columns else 0
            prec = mdf["precision"].mean()
            rec = mdf["recall"].mean()
            f1 = mdf["f1"].mean()
            cov = mdf["coverage"].mean()
            iou_m = mdf["mean_iou_matched"].mean() if "mean_iou_matched" in mdf.columns else float("nan")
            over = mdf["over_seg"].mean() if "over_seg" in mdf.columns else float("nan")
            under = mdf["under_seg"].mean() if "under_seg" in mdf.columns else float("nan")
            f.write(f"{method} & {total_gt} & {total_pred} & {prec:.3f} & {rec:.3f} "
                    f"& {f1:.3f} & {cov:.3f} & {iou_m:.3f} & {over:.3f} & {under:.3f} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved LaTeX: {tex_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["baseline", "rf"],
                        help="Methods to compare")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    )
    run_evaluation(cfg, args.methods, args.split)
