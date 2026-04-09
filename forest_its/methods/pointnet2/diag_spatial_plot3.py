"""
Seccion 3 del diagnostico: posicion espacial de puntos sin cobertura.
No requiere modelo — simula el random subsampling para obtener covered_mask
y analiza la distribucion espacial usando las datos LAS.
"""
import sys
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from forest_its.data.dataset import load_las, get_binary_labels
from forest_its.preprocessing.normalize_height import process_plot


def run():
    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
    )
    las_path = Path(cfg.paths.dataset_root) / "CULS" / "plot_3_annotated.las"
    print(f"Loading: {las_path}")

    data = load_las(las_path)
    data["_plot_stem"] = "plot_3_annotated"
    process_plot(
        data,
        resolution_dtm=cfg.preprocessing.dtm_resolution,
        smooth_window=cfg.preprocessing.smooth_window,
        hag_min=cfg.preprocessing.hag_min,
        hag_max=cfg.preprocessing.hag_max,
    )

    xyz_all = data["xyz"].astype(np.float32)
    hag_all = data["hag"].astype(np.float32)
    classification = data["classification"]
    N = len(xyz_all)
    num_points = int(cfg.pointnet2.num_points)
    n_passes = min(1000, max(50, int(N / num_points * 3)))

    print(f"N={N:,}, num_points={num_points}, n_passes={n_passes}")
    print("Simulating random subsampling coverage (no model inference)...")

    # Reproduce exact coverage tracking
    np.random.seed(0)   # fixed seed for reproducibility of this diagnostic
    prob_count = np.zeros(N, dtype=np.int32)
    for _ in range(n_passes):
        if N >= num_points:
            choice = np.random.choice(N, num_points, replace=False)
        else:
            choice = np.random.choice(N, num_points, replace=True)
        prob_count[choice] += 1

    covered_mask = prob_count > 0
    uncovered_mask = ~covered_mask
    n_uncovered = uncovered_mask.sum()

    print(f"  Covered:   {covered_mask.sum():,} ({100.*covered_mask.sum()/N:.1f}%)")
    print(f"  Uncovered: {n_uncovered:,} ({100.*n_uncovered/N:.1f}%)")

    unc_xyz = xyz_all[uncovered_mask]
    unc_hag = hag_all[uncovered_mask]
    unc_cls = classification[uncovered_mask]
    binary_labels = get_binary_labels(classification)
    unc_binary = binary_labels[uncovered_mask]

    print(f"\n{'='*60}")
    print(f"3. POSICION ESPACIAL DE PUNTOS SIN COBERTURA")
    print(f"{'='*60}")

    # --- HAG distribution ---
    print(f"\n  a) Distribucion por altura (HAG):")
    hag_bins = [(0, 0.5, "suelo/rasante"),
                (0.5, 2,  "sotobosque"),
                (2,   5,  "copa baja"),
                (5,   10, "copa media"),
                (10,  20, "copa alta"),
                (20,  50, "emergente")]
    for lo, hi, label in hag_bins:
        cnt = ((unc_hag >= lo) & (unc_hag < hi)).sum()
        pct = 100. * cnt / n_uncovered
        all_cnt = ((hag_all >= lo) & (hag_all < hi)).sum()
        ratio = 100. * cnt / all_cnt if all_cnt > 0 else 0
        print(f"    {lo:4.0f}-{hi:4.0f}m ({label:15s}): {cnt:7,} ({pct:4.1f}% de KNN)  "
              f"= {ratio:4.1f}% de todos los pts en ese rango")

    # --- Class distribution ---
    print(f"\n  b) Clase semantica de puntos KNN:")
    CLASS_NAMES = {0: "Unclassified", 1: "Low-veg", 2: "Terrain",
                   3: "Out-points", 4: "Stem", 5: "Live-branches", 6: "Woody-branches"}
    for cls_val, cls_name in CLASS_NAMES.items():
        cnt = (unc_cls == cls_val).sum()
        if cnt > 0:
            all_cls_cnt = (classification == cls_val).sum()
            ratio = 100. * cnt / all_cls_cnt if all_cls_cnt > 0 else 0
            print(f"    Class {cls_val} ({cls_name:16s}): {cnt:7,}  = {ratio:4.1f}% de esa clase en el plot")

    # --- Border vs interior ---
    print(f"\n  c) Borde vs interior del plot:")
    xy_all = xyz_all[:, :2]
    xmin, xmax = xy_all[:, 0].min(), xy_all[:, 0].max()
    ymin, ymax = xy_all[:, 1].min(), xy_all[:, 1].max()
    dx = xmax - xmin
    dy = ymax - ymin

    for margin_pct in [0.05, 0.10, 0.15]:
        border_mask = (
            (unc_xyz[:, 0] < xmin + margin_pct * dx) |
            (unc_xyz[:, 0] > xmax - margin_pct * dx) |
            (unc_xyz[:, 1] < ymin + margin_pct * dy) |
            (unc_xyz[:, 1] > ymax - margin_pct * dy)
        )
        # Expected fraction if uniform
        border_area_frac = 1 - (1 - 2*margin_pct)**2
        print(f"    Borde {margin_pct*100:.0f}% ({margin_pct*dx:.1f}m/{margin_pct*dy:.1f}m): "
              f"{border_mask.sum():7,} ({100.*border_mask.sum()/n_uncovered:.1f}%)  "
              f"[esperado si uniforme: {100.*border_area_frac:.1f}%]")

    # --- Distance to nearest covered neighbor ---
    print(f"\n  d) Distancia XY al vecino cubierto mas cercano:")
    tree_xy = cKDTree(xyz_all[covered_mask, :2])
    nn_dist, _ = tree_xy.query(unc_xyz[:, :2], k=1)
    for lo, hi in [(0,0.5),(0.5,1),(1,2),(2,5),(5,100)]:
        cnt = ((nn_dist >= lo) & (nn_dist < hi)).sum()
        print(f"    {lo:.1f}-{hi:.1f}m: {cnt:7,} ({100.*cnt/n_uncovered:.1f}%)")
    print(f"  Mean: {nn_dist.mean():.3f}m  Max: {nn_dist.max():.3f}m  "
          f"P95: {np.percentile(nn_dist, 95):.3f}m  P99: {np.percentile(nn_dist, 99):.3f}m")

    # --- Are uncovered points near tree vs non-tree covered points? ---
    print(f"\n  e) Vecino cubierto mas cercano: es arbol o no-arbol?")
    covered_binary = binary_labels[covered_mask]
    _, nn_idx = tree_xy.query(unc_xyz[:, :2], k=1)
    nn_is_tree = (covered_binary[nn_idx] == 1)
    # Only for valid (non-excluded) uncovered
    valid_unc_mask = unc_binary != -1
    print(f"    Vecino=arbol  (para KNN validos): "
          f"{nn_is_tree[valid_unc_mask].sum():,} / {valid_unc_mask.sum():,} "
          f"({100.*nn_is_tree[valid_unc_mask].mean():.1f}%)")
    print(f"    Vecino=no-arbol: "
          f"{(~nn_is_tree[valid_unc_mask]).sum():,} "
          f"({100.*(~nn_is_tree[valid_unc_mask]).mean():.1f}%)")

    print(f"\nDone.")


if __name__ == "__main__":
    run()
