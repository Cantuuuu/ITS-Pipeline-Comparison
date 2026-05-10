"""
Análisis de sensibilidad leave-one-site-out (LOSO) para métricas de instancia ITS.

Demuestra que las conclusiones principales del paper no dependen del sesgo
introducido por NIBIO (≈50% del conjunto de test).

Uso:
    python -m forest_its.evaluation.site_sensitivity
    python forest_its/evaluation/site_sensitivity.py
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

METHOD_ORDER = ["baseline", "rf", "pointnet2", "rf_density", "pointnet2_density"]
METHOD_LABELS = {
    "baseline":           "A — Baseline",
    "rf":                 "B — RF",
    "pointnet2":          "C — PN++",
    "rf_density":         "D — RF+Density",
    "pointnet2_density":  "E — PN++Density",
}
MIN_PLOTS_WARNING = 5


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _load(csv_path: Path) -> pd.DataFrame:
    """Lee el CSV y retorna las columnas relevantes."""
    df = pd.read_csv(csv_path)
    cols = ["method", "plot", "institution", "f1", "precision", "recall", "coverage"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en el CSV: {missing}")
    return df[cols].copy()


def _method_means(sub: pd.DataFrame) -> dict[str, float]:
    """Devuelve mean_f1 por method para un subconjunto de plots."""
    return sub.groupby("method")["f1"].mean().to_dict()


def _rank(means: dict[str, float]) -> list[str]:
    """Ordena methods de mayor a menor mean_f1."""
    return sorted(means, key=lambda m: means.get(m, -1), reverse=True)


# ---------------------------------------------------------------------------
# 1. Leave-one-site-out
# ---------------------------------------------------------------------------

def leave_one_site_out(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Calcula media y SD de F1 por método excluyendo un sitio a la vez.

    Para cada sitio en {CULS, NIBIO, RMIT, SCION, TUWIEN} excluye todos sus
    plots y recalcula las métricas sobre los restantes. También incluye la
    versión "full" (sin excluir ningún sitio) como referencia.

    Args:
        df: DataFrame con columnas method, plot, institution, f1, ...
        output_dir: Directorio de salida.

    Returns:
        DataFrame con columnas: excluded_site, n_plots_remaining, method,
        mean_f1, std_f1.
    """
    records = []
    sites = sorted(df["institution"].unique())
    n_plots_total = df["plot"].nunique()

    # Versión full (referencia)
    for method, grp in df.groupby("method"):
        vals = grp["f1"].values
        records.append({
            "excluded_site":    "none",
            "n_plots_remaining": n_plots_total,
            "method":           method,
            "mean_f1":          float(np.mean(vals)),
            "std_f1":           float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
        })

    # LOSO
    for site in sites:
        sub = df[df["institution"] != site]
        n_remaining = sub["plot"].nunique()

        if n_remaining < MIN_PLOTS_WARNING:
            warnings.warn(
                f"Al excluir {site} quedan solo {n_remaining} plots — "
                "las estimaciones pueden ser inestables.",
                UserWarning,
                stacklevel=2,
            )

        for method, grp in sub.groupby("method"):
            vals = grp["f1"].values
            records.append({
                "excluded_site":     site,
                "n_plots_remaining": n_remaining,
                "method":            method,
                "mean_f1":           float(np.mean(vals)),
                "std_f1":            float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            })

    result = pd.DataFrame(records)
    out_path = output_dir / "leave_one_site_out.csv"
    result.to_csv(out_path, index=False)
    return result


# ---------------------------------------------------------------------------
# 2. Estabilidad del ranking
# ---------------------------------------------------------------------------

def ranking_stability(loso_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Para cada exclusión de sitio determina el ranking de métodos por mean_f1
    y compara con el ranking "full".

    Args:
        loso_df: DataFrame producido por leave_one_site_out.
        output_dir: Directorio de salida.

    Returns:
        DataFrame con: excluded_site, rank_1..rank_5, ranking_changed.
    """
    full_means = (
        loso_df[loso_df["excluded_site"] == "none"]
        .set_index("method")["mean_f1"]
        .to_dict()
    )
    full_rank = _rank(full_means)

    records = []
    for excl_site in loso_df["excluded_site"].unique():
        sub = loso_df[loso_df["excluded_site"] == excl_site]
        means = sub.set_index("method")["mean_f1"].to_dict()
        ranked = _rank(means)

        # Rellenar hasta 5 posiciones (por si algún método no tiene datos)
        ranked_padded = ranked + ["—"] * (5 - len(ranked))

        # El ranking cambió si el orden de los métodos presentes en ambos difiere
        common = [m for m in full_rank if m in means]
        full_common = [m for m in full_rank if m in common]
        loso_common = [m for m in ranked if m in common]
        changed = full_common != loso_common

        records.append({
            "excluded_site":   excl_site,
            "n_plots":         int(sub["n_plots_remaining"].iloc[0]),
            "rank_1":          ranked_padded[0],
            "rank_2":          ranked_padded[1],
            "rank_3":          ranked_padded[2],
            "rank_4":          ranked_padded[3],
            "rank_5":          ranked_padded[4],
            "ranking_changed": changed,
        })

    result = pd.DataFrame(records)
    out_path = output_dir / "ranking_stability.csv"
    result.to_csv(out_path, index=False)
    return result


# ---------------------------------------------------------------------------
# 3. Impacto de NIBIO
# ---------------------------------------------------------------------------

def nibio_impact(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Cuantifica el sesgo introducido por NIBIO comparando métricas con y sin
    ese sitio, y verifica que las conclusiones clave del paper se mantengan.

    Conclusiones clave verificadas:
        1. RF+Density sigue siendo el mejor flujo sin NIBIO.
        2. Density seeding sigue superando a CHM sin NIBIO
           (rf_density > rf  y  pointnet2_density > pointnet2).
        3. RF sigue superando a PointNet++ sin NIBIO
           (rf > pointnet2  y  rf_density > pointnet2_density).

    Args:
        df: DataFrame con columnas method, plot, institution, f1, ...
        output_dir: Directorio de salida.

    Returns:
        DataFrame con: method, f1_with_nibio, f1_without_nibio, delta, pct_change.
    """
    df_with    = df
    df_without = df[df["institution"] != "NIBIO"]

    n_with    = df_with["plot"].nunique()
    n_without = df_without["plot"].nunique()

    records = []
    for method in METHOD_ORDER:
        v_with    = df_with[df_with["method"] == method]["f1"].values
        v_without = df_without[df_without["method"] == method]["f1"].values

        f1_with    = float(np.mean(v_with))    if len(v_with)    > 0 else float("nan")
        f1_without = float(np.mean(v_without)) if len(v_without) > 0 else float("nan")
        delta      = f1_without - f1_with
        pct_change = 100.0 * delta / f1_with if f1_with != 0 else float("nan")

        records.append({
            "method":         method,
            "f1_with_nibio":    round(f1_with, 4),
            "f1_without_nibio": round(f1_without, 4),
            "delta":            round(delta, 4),
            "pct_change":       round(pct_change, 1),
            "n_plots_with":     n_with,
            "n_plots_without":  n_without,
        })

    result = pd.DataFrame(records)
    out_path = output_dir / "nibio_impact.csv"
    result.to_csv(out_path, index=False)
    return result


def _check_conclusions(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    """
    Verifica las tres conclusiones clave con y sin NIBIO.

    Returns:
        Lista de (descripción, se_mantiene, detalle).
    """
    conclusions = []

    for label, sub in [("CON NIBIO", df), ("SIN NIBIO", df[df["institution"] != "NIBIO"])]:
        means = sub.groupby("method")["f1"].mean().to_dict()
        best  = max(means, key=lambda m: means[m])

        c1 = best == "rf_density"
        conclusions.append((
            f"[{label}] RF+Density es el mejor flujo",
            c1,
            f"mejor={best} (mean_f1={means.get(best, 0):.4f})",
        ))

        c2a = means.get("rf_density", 0) > means.get("rf", 0)
        c2b = means.get("pointnet2_density", 0) > means.get("pointnet2", 0)
        conclusions.append((
            f"[{label}] Density seeding supera a CHM (RF: {c2a}, PN++: {c2b})",
            c2a and c2b,
            f"rf_density={means.get('rf_density',0):.4f} vs rf={means.get('rf',0):.4f} | "
            f"pn++_density={means.get('pointnet2_density',0):.4f} vs pn++={means.get('pointnet2',0):.4f}",
        ))

        c3a = means.get("rf", 0) > means.get("pointnet2", 0)
        c3b = means.get("rf_density", 0) > means.get("pointnet2_density", 0)
        conclusions.append((
            f"[{label}] RF supera a PointNet++ (CHM: {c3a}, Density: {c3b})",
            c3a and c3b,
            f"rf={means.get('rf',0):.4f} vs pn++={means.get('pointnet2',0):.4f} | "
            f"rf_density={means.get('rf_density',0):.4f} vs pn++_density={means.get('pointnet2_density',0):.4f}",
        ))

    return conclusions


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orquesta la carga, análisis LOSO y exportación de resultados."""
    parser = argparse.ArgumentParser(
        description="Análisis leave-one-site-out para métricas de instancia ITS."
    )
    parser.add_argument(
        "--input",
        default="results/comparison_table_test.csv",
        help="Ruta al comparison_table_test.csv (default: results/comparison_table_test.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directorio de salida (default: results/)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] No se encuentra: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Leyendo: {input_path}")
    df = _load(input_path)
    sites    = sorted(df["institution"].unique())
    n_plots  = df["plot"].nunique()
    print(f"Sitios:  {sites}")
    print(f"Plots:   {n_plots}")
    nibio_n  = df[df["institution"] == "NIBIO"]["plot"].nunique()
    print(f"NIBIO representa {nibio_n}/{n_plots} plots ({100*nibio_n/n_plots:.0f}% del test)")
    print()

    # --- LOSO ---
    print("=" * 70)
    print("LEAVE-ONE-SITE-OUT — mean F1 por método")
    print("=" * 70)
    loso = leave_one_site_out(df, output_dir)

    pivot = loso.pivot_table(
        index="excluded_site", columns="method", values="mean_f1"
    )
    col_order = [m for m in METHOD_ORDER if m in pivot.columns]
    print(pivot[col_order].round(4).to_string())
    print(f"\n→ Exportado: {output_dir}/leave_one_site_out.csv")
    print()

    # --- Estabilidad del ranking ---
    print("=" * 70)
    print("ESTABILIDAD DEL RANKING")
    print("=" * 70)
    stab = ranking_stability(loso, output_dir)
    for _, row in stab.iterrows():
        changed_str = "⚠ CAMBIÓ" if row["ranking_changed"] else "✓ estable"
        print(
            f"  excluir={row['excluded_site']:<8}  "
            f"[{row['rank_1']} > {row['rank_2']} > {row['rank_3']} > "
            f"{row['rank_4']} > {row['rank_5']}]  {changed_str}"
        )
    print(f"\n→ Exportado: {output_dir}/ranking_stability.csv")
    print()

    # --- Impacto de NIBIO ---
    print("=" * 70)
    print("IMPACTO DE NIBIO — con vs sin NIBIO")
    print("=" * 70)
    impact = nibio_impact(df, output_dir)
    for _, row in impact.iterrows():
        direction = "↑" if row["delta"] > 0 else ("↓" if row["delta"] < 0 else "=")
        print(
            f"  {row['method']:<22}  "
            f"con={row['f1_with_nibio']:.4f}  "
            f"sin={row['f1_without_nibio']:.4f}  "
            f"Δ={row['delta']:+.4f} {direction}  "
            f"({row['pct_change']:+.1f}%)"
        )
    print(f"\n→ Exportado: {output_dir}/nibio_impact.csv")
    print()

    # --- Verificación de conclusiones ---
    print("=" * 70)
    print("VERIFICACIÓN DE CONCLUSIONES CLAVE")
    print("=" * 70)
    conclusions = _check_conclusions(df)
    all_maintained = True
    for desc, ok, detail in conclusions:
        status = "✓ SE MANTIENE" if ok else "✗ NO SE MANTIENE"
        print(f"  {status}  {desc}")
        print(f"           {detail}")
        if not ok:
            all_maintained = False
    print()

    # --- Resumen por sitio excluido ---
    print("=" * 70)
    print("RESUMEN: ¿Las conclusiones se mantienen al excluir cada sitio?")
    print("=" * 70)
    for excl_site in ["none"] + list(sites):
        sub = df if excl_site == "none" else df[df["institution"] != excl_site]
        means = sub.groupby("method")["f1"].mean().to_dict()
        best  = max(means, key=lambda m: means[m])
        rf_best  = best == "rf_density"
        dens_ok  = (means.get("rf_density", 0) > means.get("rf", 0) and
                    means.get("pointnet2_density", 0) > means.get("pointnet2", 0))
        rf_ok    = (means.get("rf", 0) > means.get("pointnet2", 0) and
                    means.get("rf_density", 0) > means.get("pointnet2_density", 0))
        all_ok   = rf_best and dens_ok and rf_ok
        verdict  = "✓ Las conclusiones SE MANTIENEN" if all_ok else "✗ Alguna conclusión NO SE MANTIENE"
        excl_lbl = f"excluir {excl_site}" if excl_site != "none" else "full (sin exclusión)"
        print(f"  {excl_lbl:<20}  {verdict}  [mejor={best}]")

    print()
    print("Completado.")


if __name__ == "__main__":
    main()
