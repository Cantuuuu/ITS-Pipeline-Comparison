"""
Pruebas estadísticas para comparación de pipelines ITS.

Lee results/comparison_table_test.csv, calcula estadísticas descriptivas,
pruebas de permutación exacta (principal) y Wilcoxon (referencia), bootstrap
BCa CIs, y exporta los resultados a results/.

Uso:
    python -m forest_its.evaluation.statistical_tests
    python forest_its/evaluation/statistical_tests.py
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

METRICS = ["f1", "precision", "recall", "coverage"]

COMPARISONS = [
    ("rf",            "rf_density",        "B_vs_D: RF vs RF+Density"),
    ("pointnet2",     "pointnet2_density", "C_vs_E: PN++ vs PN++Density"),
    ("baseline",      "rf_density",        "A_vs_D: Baseline vs RF+Density"),
    ("rf",            "pointnet2",         "B_vs_C: RF vs PN++"),
    ("rf_density",    "pointnet2_density", "D_vs_E: RF+Density vs PN++Density"),
]


# ---------------------------------------------------------------------------
# 1. Carga
# ---------------------------------------------------------------------------

def load_metrics(csv_path: str | Path) -> pd.DataFrame:
    """
    Lee el CSV de comparación y retorna las columnas relevantes.

    Args:
        csv_path: Ruta al comparison_table_test.csv.

    Returns:
        DataFrame con columnas: method, plot, institution, f1, precision,
        recall, coverage.
    """
    df = pd.read_csv(csv_path)
    cols = ["method", "plot", "institution", "f1", "precision", "recall", "coverage"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en el CSV: {missing}")
    return df[cols].copy()


# ---------------------------------------------------------------------------
# 2. Estadísticas descriptivas
# ---------------------------------------------------------------------------

def descriptive_stats(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas por método para f1, precision, recall
    y coverage.

    Args:
        df: DataFrame cargado por load_metrics.
        output_dir: Directorio donde exportar descriptive_stats_test.csv.

    Returns:
        DataFrame con mean, std, median, min, max por method y métrica.
    """
    records = []
    for method, grp in df.groupby("method"):
        for metric in METRICS:
            vals = grp[metric].values
            records.append({
                "method":  method,
                "metric":  metric,
                "mean":    float(np.mean(vals)),
                "std":     float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                "median":  float(np.median(vals)),
                "min":     float(np.min(vals)),
                "max":     float(np.max(vals)),
                "n":       len(vals),
            })

    result = pd.DataFrame(records)
    out_path = output_dir / "descriptive_stats_test.csv"
    result.to_csv(out_path, index=False)
    return result


# ---------------------------------------------------------------------------
# 3a. Test de permutación exacto (prueba PRINCIPAL)
# ---------------------------------------------------------------------------

def exact_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided",
) -> dict:
    """
    Test de permutación exacto sobre diferencias pareadas d = y - x.

    Enumera todas las 2^n asignaciones de signo posibles y calcula la
    distribución exacta de la media de las diferencias bajo la hipótesis
    nula de intercambiabilidad. Con n=11 se generan 2048 permutaciones.

    Args:
        x: Valores del método A (array de longitud n).
        y: Valores del método B (array de longitud n).
        alternative: 'two-sided' (bilateral).

    Returns:
        Dict con: observed_mean_diff, p_value, n_permutations, n_pairs.
    """
    d = y - x
    n = len(d)
    observed = float(np.mean(d))

    # Generar todas las 2^n asignaciones de signo: (+1, -1)
    signs = list(itertools.product([-1, 1], repeat=n))
    n_perm = len(signs)  # 2^n

    perm_means = np.array([
        np.mean(np.array(s) * d) for s in signs
    ])

    if alternative == "two-sided":
        p_value = float(np.mean(np.abs(perm_means) >= np.abs(observed)))
    elif alternative == "greater":
        p_value = float(np.mean(perm_means >= observed))
    else:
        p_value = float(np.mean(perm_means <= observed))

    return {
        "observed_mean_diff": observed,
        "p_value":            p_value,
        "n_permutations":     n_perm,
        "n_pairs":            n,
    }


# ---------------------------------------------------------------------------
# 3b. Wilcoxon (referencia, _legacy)
# ---------------------------------------------------------------------------

def _wilcoxon_legacy(
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    n_nonzero: int,
) -> tuple[float, float, str]:
    """
    Ejecuta Wilcoxon signed-rank exact como referencia secundaria.

    Returns:
        (W, p_value, note)
    """
    if n_nonzero < 2:
        return float("nan"), float("nan"), "insuficientes_pares_no_nulos"
    try:
        res = stats.wilcoxon(a_vals, b_vals, alternative="two-sided", method="exact")
        return float(res.statistic), float(res.pvalue), ""
    except Exception as exc:
        return float("nan"), float("nan"), str(exc)


# ---------------------------------------------------------------------------
# 3c. paired_tests (reemplaza wilcoxon_tests)
# ---------------------------------------------------------------------------

def paired_tests(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Ejecuta pruebas pareadas para cada comparación de métodos:
      - Test de permutación exacto (PRINCIPAL): p_perm
      - Wilcoxon signed-rank exact (referencia): p_wilcoxon

    Comparaciones:
        B vs D: rf vs rf_density
        C vs E: pointnet2 vs pointnet2_density
        A vs D: baseline vs rf_density
        B vs C: rf vs pointnet2
        D vs E: rf_density vs pointnet2_density

    Args:
        df: DataFrame cargado por load_metrics.
        output_dir: Directorio donde exportar paired_tests.csv.

    Returns:
        DataFrame con ambos p-values y estadísticos por comparación y métrica.
    """
    records = []

    for m_a, m_b, label in COMPARISONS:
        sub_a = df[df["method"] == m_a].set_index("plot")
        sub_b = df[df["method"] == m_b].set_index("plot")
        common_plots = sub_a.index.intersection(sub_b.index)

        for metric in ["f1", "precision", "recall"]:
            a_vals = sub_a.loc[common_plots, metric].values
            b_vals = sub_b.loc[common_plots, metric].values
            diffs  = b_vals - a_vals

            nonzero_mask = diffs != 0
            n_nonzero    = int(nonzero_mask.sum())
            mean_diff    = float(np.mean(diffs))
            std_diff     = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")

            # -- Permutación exacta (principal) --
            perm = exact_permutation_test(a_vals, b_vals, alternative="two-sided")

            # -- Wilcoxon (referencia) --
            w_stat, p_wilcoxon, note = _wilcoxon_legacy(a_vals, b_vals, n_nonzero)

            records.append({
                "comparison":        label,
                "method_a":          m_a,
                "method_b":          m_b,
                "metric":            metric,
                "n_plots":           len(common_plots),
                "n_nonzero_pairs":   n_nonzero,
                "mean_diff":         mean_diff,
                "std_diff":          std_diff,
                # Permutación (principal)
                "perm_observed_diff": perm["observed_mean_diff"],
                "perm_p_value":       perm["p_value"],
                "n_permutations":     perm["n_permutations"],
                # Wilcoxon (referencia)
                "W":                  w_stat,
                "p_wilcoxon":         p_wilcoxon,
                "note":               note,
            })

    result = pd.DataFrame(records)
    out_path = output_dir / "paired_tests.csv"
    result.to_csv(out_path, index=False)
    return result


# ---------------------------------------------------------------------------
# 3d. wilcoxon_tests (legacy — mantener para compatibilidad)
# ---------------------------------------------------------------------------

def wilcoxon_tests(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    [LEGACY] Ejecuta solo Wilcoxon signed-rank tests.
    Usar paired_tests() para el análisis completo con permutación exacta.

    Exporta a results/wilcoxon_tests.csv.
    """
    records = []

    for m_a, m_b, label in COMPARISONS:
        sub_a = df[df["method"] == m_a].set_index("plot")
        sub_b = df[df["method"] == m_b].set_index("plot")
        common_plots = sub_a.index.intersection(sub_b.index)

        for metric in ["f1", "precision", "recall"]:
            a_vals = sub_a.loc[common_plots, metric].values
            b_vals = sub_b.loc[common_plots, metric].values
            diffs  = b_vals - a_vals

            n_nonzero = int((diffs != 0).sum())
            mean_diff = float(np.mean(diffs))
            std_diff  = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")

            w_stat, p_val, note = _wilcoxon_legacy(a_vals, b_vals, n_nonzero)

            records.append({
                "comparison":      label,
                "method_a":        m_a,
                "method_b":        m_b,
                "metric":          metric,
                "n_plots":         len(common_plots),
                "n_nonzero_pairs": n_nonzero,
                "W":               w_stat,
                "p_value":         p_val,
                "mean_diff":       mean_diff,
                "std_diff":        std_diff,
                "note":            note,
            })

    result = pd.DataFrame(records)
    out_path = output_dir / "wilcoxon_tests.csv"
    result.to_csv(out_path, index=False)
    return result


# ---------------------------------------------------------------------------
# 4a. Bootstrap BCa CI (reemplaza percentil simple)
# ---------------------------------------------------------------------------

def bootstrap_bca_ci(
    data: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """
    Bootstrap BCa (bias-corrected and accelerated) CI para la media.

    Calcula la corrección de sesgo (z0) y aceleración (a_hat) mediante
    jackknife, ajusta los percentiles y devuelve el intervalo corregido.

    Args:
        data: Array de observaciones.
        n_boot: Número de muestras bootstrap.
        ci: Nivel de confianza (0.95 → IC 95%).
        seed: Semilla para reproducibilidad.

    Returns:
        (ci_lower, ci_upper, boot_means_array)
    """
    rng  = np.random.default_rng(seed)
    n    = len(data)
    obs  = float(np.mean(data))

    boot_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_boot)
    ])

    # Corrección de sesgo z0: proporción de boot means < obs
    prop_less = float(np.mean(boot_means < obs))
    # Proteger contra prop_less == 0 o 1 (llevaría a inf en ppf)
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
    z0 = float(stats.norm.ppf(prop_less))

    # Aceleración a_hat via jackknife
    jack_means = np.array([
        np.mean(np.delete(data, i)) for i in range(n)
    ])
    jack_mean  = float(np.mean(jack_means))
    num   = np.sum((jack_mean - jack_means) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack_means) ** 2) ** 1.5)
    a_hat = float(num / denom) if denom != 0 else 0.0

    alpha   = 1.0 - ci
    z_alpha = float(stats.norm.ppf(alpha / 2))        # negativo
    z_1ma   = float(stats.norm.ppf(1.0 - alpha / 2))  # positivo

    def _adj_alpha(z_a: float) -> float:
        denom_val = 1.0 - a_hat * (z0 + z_a)
        if denom_val == 0:
            return z_a
        return float(stats.norm.cdf(z0 + (z0 + z_a) / denom_val))

    alpha1 = _adj_alpha(z_alpha)
    alpha2 = _adj_alpha(z_1ma)

    ci_lower = float(np.percentile(boot_means, 100.0 * alpha1))
    ci_upper = float(np.percentile(boot_means, 100.0 * alpha2))

    return ci_lower, ci_upper, boot_means


# ---------------------------------------------------------------------------
# 4b. bootstrap_ci actualizado (usa BCa internamente)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    df: pd.DataFrame,
    output_dir: Path,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Calcula bootstrap BCa CIs para la media de f1, precision, recall y
    coverage por método, y para las diferencias pareadas de f1.

    Args:
        df: DataFrame cargado por load_metrics.
        output_dir: Directorio donde exportar bootstrap_ci.csv.
        n_boot: Número de muestras bootstrap.
        ci: Nivel de confianza (0.95 → IC 95%).
        seed: Semilla para reproducibilidad.

    Returns:
        DataFrame con observed, ci_lower, ci_upper, ci_method="BCa" por
        method/comparación y métrica.
    """
    records = []
    # Semillas derivadas determinísticamente para que cada llamada a
    # bootstrap_bca_ci sea reproducible y no se solapen entre sí.
    rng_seeds = iter(range(seed, seed + 10_000))

    # -- BCa CIs de la media por método --
    for method, grp in df.groupby("method"):
        for metric in METRICS:
            vals = grp[metric].values
            if len(vals) == 0:
                continue
            lo, hi, _ = bootstrap_bca_ci(vals, n_boot=n_boot, ci=ci, seed=next(rng_seeds))
            records.append({
                "type":       "method_mean",
                "label":      method,
                "metric":     metric,
                "observed":   float(np.mean(vals)),
                "ci_lower":   lo,
                "ci_upper":   hi,
                "n":          len(vals),
                "n_boot":     n_boot,
                "ci_level":   ci,
                "ci_method":  "BCa",
            })

    # -- BCa CIs de diferencias pareadas de f1 --
    for m_a, m_b, label in COMPARISONS:
        sub_a = df[df["method"] == m_a].set_index("plot")
        sub_b = df[df["method"] == m_b].set_index("plot")
        common_plots = sub_a.index.intersection(sub_b.index)

        if len(common_plots) == 0:
            continue

        diffs = (
            sub_b.loc[common_plots, "f1"].values
            - sub_a.loc[common_plots, "f1"].values
        )
        lo, hi, _ = bootstrap_bca_ci(diffs, n_boot=n_boot, ci=ci, seed=next(rng_seeds))
        records.append({
            "type":       "paired_diff_f1",
            "label":      label,
            "metric":     "f1_diff",
            "observed":   float(np.mean(diffs)),
            "ci_lower":   lo,
            "ci_upper":   hi,
            "n":          len(diffs),
            "n_boot":     n_boot,
            "ci_level":   ci,
            "ci_method":  "BCa",
        })

    result = pd.DataFrame(records)
    out_path = output_dir / "bootstrap_ci.csv"
    result.to_csv(out_path, index=False)
    return result


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def _sig(p: float) -> str:
    """Convierte p-value en símbolo de significancia."""
    if np.isnan(p):
        return "N/A"
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "ns"))


def main() -> None:
    """Orquesta la carga, cálculo y exportación de estadísticas."""
    parser = argparse.ArgumentParser(
        description="Pruebas estadísticas para comparación de pipelines ITS."
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
    df = load_metrics(input_path)
    methods = sorted(df["method"].unique())
    print(f"Métodos encontrados: {methods}")
    print(f"Plots encontrados:   {df['plot'].nunique()} ({df['plot'].unique().tolist()})")
    print()

    # --- Descriptivas ---
    print("=" * 75)
    print("ESTADÍSTICAS DESCRIPTIVAS (F1)")
    print("=" * 75)
    desc = descriptive_stats(df, output_dir)
    f1_desc = (
        desc[desc["metric"] == "f1"]
        .set_index("method")[["mean", "std", "median", "min", "max"]]
    )
    print(f1_desc.round(4).to_string())
    print(f"\n→ Exportado: {output_dir}/descriptive_stats_test.csv")
    print()

    # --- Paired tests (permutación + Wilcoxon) ---
    print("=" * 75)
    print("PRUEBAS PAREADAS — F1 (n_permutaciones=2048, Wilcoxon exact)")
    print(f"  {'Comparación':<42} {'p_perm':>8}  {'p_wilcox':>9}  {'Δmean':>7}  sig_perm")
    print("-" * 75)
    pt = paired_tests(df, output_dir)
    f1_pt = pt[pt["metric"] == "f1"]
    for _, row in f1_pt.iterrows():
        p_perm  = row["perm_p_value"]
        p_wilcox = row["p_wilcoxon"]
        p_perm_str   = f"{p_perm:.4f}"   if not np.isnan(p_perm)   else "  N/A "
        p_wilcox_str = f"{p_wilcox:.4f}" if not np.isnan(p_wilcox) else "   N/A"
        sig = _sig(p_perm)
        print(
            f"  {row['comparison']:<42} {p_perm_str:>8}  {p_wilcox_str:>9}  "
            f"{row['mean_diff']:>+7.4f}  {sig}"
        )
    print()
    print("  Significancia (test de permutación): *** p<0.01  ** p<0.05  * p<0.10  ns")
    print(f"\n→ Exportado: {output_dir}/paired_tests.csv")
    print()

    # --- Bootstrap BCa CIs ---
    print("=" * 75)
    print("BOOTSTRAP BCa CIs 95% — MEDIA DE F1 POR MÉTODO (n_boot=10,000)")
    print("=" * 75)
    boot = bootstrap_ci(df, output_dir)
    method_boot = boot[(boot["type"] == "method_mean") & (boot["metric"] == "f1")]
    for _, row in method_boot.iterrows():
        print(
            f"  {row['label']:<22}  mean={row['observed']:.4f}  "
            f"BCa 95% CI [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
        )
    print()
    print("BCa CIs 95% — DIFERENCIAS PAREADAS DE F1")
    print("-" * 75)
    diff_boot = boot[boot["type"] == "paired_diff_f1"]
    for _, row in diff_boot.iterrows():
        sign      = "↑" if row["observed"] > 0 else "↓"
        zero_excl = "✓ CI excluye 0" if (row["ci_lower"] > 0 or row["ci_upper"] < 0) else "  CI incluye 0"
        print(
            f"  {row['label']:<42}  Δ={row['observed']:+.4f} {sign}  "
            f"BCa [{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}]  {zero_excl}"
        )
    print(f"\n→ Exportado: {output_dir}/bootstrap_ci.csv")
    print()

    # --- Conclusiones por comparación ---
    print("=" * 75)
    print("CONCLUSIÓN: significancia con test de permutación exacto (F1)")
    print("=" * 75)
    for _, row in f1_pt.iterrows():
        p = row["perm_p_value"]
        sig = _sig(p)
        if sig in ("**", "***"):
            verdict = f"SIGNIFICATIVA (p={p:.4f})"
        elif sig == "*":
            verdict = f"MARGINAL (p={p:.4f})"
        else:
            p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
            verdict = f"no significativa (p={p_str})"
        direction = "B>A" if row["mean_diff"] > 0 else "A>B"
        print(f"  {row['comparison']:<42}  {verdict}  [{direction}]")
    print()
    print("Completado.")


if __name__ == "__main__":
    main()
