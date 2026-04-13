"""
Genera figuras visuales para el paper usando datos reales de predicciones y LAS.
Salida: assets/paper/  (tracked en git)
Ejecutar desde raiz del proyecto: python generate_visual_figures.py
"""

import zipfile, io, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT = Path("assets/paper")
OUT.mkdir(parents=True, exist_ok=True)
PREDS_ZIP = "C:/Users/cantu/Downloads/predictions.zip"
LAS_DIR   = "C:/Users/cantu/Downloads/FORinstance_dataset"
RESULTS   = "results/comparison_table_test.csv"

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    print("[WARN] laspy not found, skipping LAS-based figures")

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10, "figure.dpi": 150})

METHODS  = ["baseline", "rf", "pointnet2", "rf_density", "pointnet2_density"]
M_LABELS = {
    "baseline":           "A: Sin preproc.",
    "rf":                 "B: RF + CHM",
    "pointnet2":          "C: PN++ + CHM",
    "rf_density":         "D: RF + Density",
    "pointnet2_density":  "E: PN++ + Density",
}
M_COLORS = {
    "baseline":          "#7F8C8D",
    "rf":                "#2980B9",
    "pointnet2":         "#8E44AD",
    "rf_density":        "#E74C3C",
    "pointnet2_density": "#C0392B",
}

# Paleta discreta para instancias (hasta 30 árboles)
CMAP_INST = plt.cm.get_cmap("tab20b", 25)

def load_npz(method, plot_name):
    """Carga predictions/method/plot_name_instances.npz desde ZIP."""
    fname = f"predictions/{method}/{plot_name}_instances.npz"
    with zipfile.ZipFile(PREDS_ZIP) as z:
        if fname not in z.namelist():
            return None
        with z.open(fname) as f:
            return np.load(io.BytesIO(f.read()), allow_pickle=True)

def load_las(institution, plot_name):
    """Carga el LAS de GT."""
    p = f"{LAS_DIR}/{institution}/{plot_name}.las"
    return laspy.read(p) if HAS_LASPY else None

# ─────────────────────────────────────────────────────────────────────────────
# FIG-V1: Vista superior GT vs predicciones (CULS plot_2 — mejor caso)
#         5 paneles: GT + 4 métodos seleccionados
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_v1_topview_culs...")
if HAS_LASPY:
    las = load_las("CULS", "plot_2_annotated")
    x_all = np.array(las.x)
    y_all = np.array(las.y)
    z_all = np.array(las.z)
    gt_ids = np.array(las["treeID"]).astype(int)

    show_methods = ["baseline", "rf", "rf_density", "pointnet2_density"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), sharex=True, sharey=True)
    fig.suptitle("Instance Segmentation — CULS (Temperate Conifer, Best-Case Forest)",
                 fontsize=12, fontweight="bold", y=1.01)

    # Panel GT
    ax = axes[0]
    unique_gt = [g for g in np.unique(gt_ids) if g > 0]
    for i, gid in enumerate(unique_gt):
        mask = gt_ids == gid
        ax.scatter(x_all[mask], y_all[mask], s=0.15,
                   color=CMAP_INST(i % 25), rasterized=True)
    mask0 = gt_ids == 0
    ax.scatter(x_all[mask0], y_all[mask0], s=0.05, color="#CCCCCC",
               alpha=0.3, rasterized=True)
    ax.set_title("Ground Truth", fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.02, 0.02, f"{len(unique_gt)} trees", transform=ax.transAxes,
            fontsize=8, color="black", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Paneles predicciones
    for ax, method in zip(axes[1:], show_methods):
        data = load_npz(method, "CULS__plot_2_annotated")
        if data is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(M_LABELS[method])
            continue
        pred_ids = data["instance_ids"]
        unique_pred = [p for p in np.unique(pred_ids) if p > 0]

        # Sort by size descending so biggest trees dominate color
        sizes = [(pid, (pred_ids == pid).sum()) for pid in unique_pred]
        sizes.sort(key=lambda s: -s[1])

        for i, (pid, _) in enumerate(sizes):
            mask = pred_ids == pid
            ax.scatter(x_all[mask], y_all[mask], s=0.15,
                       color=CMAP_INST(i % 25), rasterized=True)
        mask0 = pred_ids == 0
        ax.scatter(x_all[mask0], y_all[mask0], s=0.05,
                   color="#CCCCCC", alpha=0.3, rasterized=True)

        # Obtener F1 de CSV
        df = pd.read_csv(RESULTS)
        row = df[(df["method"] == method) &
                 (df["plot"].str.contains("CULS__plot_2"))]
        f1 = row["f1"].values[0] if len(row) else 0.0
        prec = row["precision"].values[0] if len(row) else 0.0
        rec  = row["recall"].values[0] if len(row) else 0.0

        ax.set_title(M_LABELS[method], fontweight="bold",
                     color=M_COLORS[method])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.02, 0.02,
                f"P={prec:.2f}  R={rec:.2f}\nF1={f1:.3f}  n={len(unique_pred)}",
                transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUT / "fig_v1_topview_culs.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("[OK] fig_v1_topview_culs.png")
    plt.close()
else:
    print("[SKIP] fig_v1 — laspy not available")

# ─────────────────────────────────────────────────────────────────────────────
# FIG-V2: Vista superior NIBIO (bosque boreal — caso difícil)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_v2_topview_nibio...")
if HAS_LASPY:
    las = load_las("NIBIO", "plot_17_annotated")
    x_all = np.array(las.x)
    y_all = np.array(las.y)
    gt_ids = np.array(las["treeID"]).astype(int)

    show_methods = ["baseline", "rf", "rf_density", "pointnet2_density"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), sharex=True, sharey=True)
    fig.suptitle("Instance Segmentation — NIBIO (Boreal Conifer, Dense Canopy)",
                 fontsize=12, fontweight="bold", y=1.01)

    ax = axes[0]
    unique_gt = [g for g in np.unique(gt_ids) if g > 0]
    for i, gid in enumerate(unique_gt):
        mask = gt_ids == gid
        ax.scatter(x_all[mask], y_all[mask], s=0.1,
                   color=CMAP_INST(i % 25), rasterized=True)
    mask0 = gt_ids == 0
    ax.scatter(x_all[mask0], y_all[mask0], s=0.04,
               color="#CCCCCC", alpha=0.2, rasterized=True)
    ax.set_title("Ground Truth", fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.02, 0.02, f"{len(unique_gt)} trees", transform=ax.transAxes,
            fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    for ax, method in zip(axes[1:], show_methods):
        data = load_npz(method, "NIBIO__plot_17_annotated")
        if data is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        pred_ids = data["instance_ids"]
        unique_pred = [p for p in np.unique(pred_ids) if p > 0]
        sizes = [(pid, (pred_ids == pid).sum()) for pid in unique_pred]
        sizes.sort(key=lambda s: -s[1])
        for i, (pid, _) in enumerate(sizes):
            mask = pred_ids == pid
            ax.scatter(x_all[mask], y_all[mask], s=0.1,
                       color=CMAP_INST(i % 25), rasterized=True)
        mask0 = pred_ids == 0
        ax.scatter(x_all[mask0], y_all[mask0], s=0.04,
                   color="#CCCCCC", alpha=0.2, rasterized=True)

        df = pd.read_csv(RESULTS)
        row = df[(df["method"] == method) &
                 (df["plot"].str.contains("NIBIO__plot_17"))]
        f1 = row["f1"].values[0] if len(row) else 0.0
        prec = row["precision"].values[0] if len(row) else 0.0
        rec  = row["recall"].values[0] if len(row) else 0.0

        ax.set_title(M_LABELS[method], fontweight="bold",
                     color=M_COLORS[method])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.02, 0.02,
                f"P={prec:.2f}  R={rec:.2f}\nF1={f1:.3f}  n={len(unique_pred)}",
                transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUT / "fig_v2_topview_nibio.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("[OK] fig_v2_topview_nibio.png")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG-V3: Semantic segmentation overlay — CULS (RF vs PN++ vs GT)
#         Muestra dónde falla cada clasificador semántico
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_v3_semantic_overlay...")
if HAS_LASPY:
    las = load_las("CULS", "plot_2_annotated")
    x_all = np.array(las.x)
    y_all = np.array(las.y)
    clsf  = np.array(las.classification).astype(int)
    # GT: tree = class 4,5,6 (branches), non-tree = 2 (ground), 3 (outpoint)
    gt_sem = np.zeros(len(x_all), dtype=int)
    gt_sem[(clsf == 4) | (clsf == 5) | (clsf == 6)] = 1  # tree

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    fig.suptitle("Semantic Segmentation Comparison — CULS Plot 2",
                 fontsize=12, fontweight="bold")

    sem_colors = {-1: "#DDDDDD", 0: "#95A5A6", 1: "#27AE60"}
    sem_labels = {-1: "Unclassified", 0: "Non-tree", 1: "Tree"}

    def plot_sem(ax, sem, title, color_override=None):
        for val in [-1, 0, 1]:
            mask = sem == val
            if mask.sum() == 0:
                continue
            col = color_override.get(val, sem_colors[val]) if color_override else sem_colors[val]
            ax.scatter(x_all[mask], y_all[mask], s=0.08,
                       color=col, alpha=0.7, rasterized=True)
        ax.set_title(title, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")
        n_tree = (sem == 1).sum()
        n_tot  = (sem != -1).sum()
        pct = 100 * n_tree / n_tot if n_tot > 0 else 0
        ax.text(0.02, 0.02, f"Tree: {pct:.1f}%",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    plot_sem(axes[0], gt_sem, "Ground Truth")

    for ax, method, title in [
        (axes[1], "rf",       "B: Random Forest"),
        (axes[2], "pointnet2","C: PointNet++ MSG"),
    ]:
        data = load_npz(method, "CULS__plot_2_annotated")
        if data is None:
            continue
        sem = data["semantic_pred"]
        # Color errors differently
        color_override = {-1: "#DDDDDD", 0: "#95A5A6", 1: "#27AE60"}
        # Mark false positives (pred=tree, gt=non-tree) in red
        fp_mask = (sem == 1) & (gt_sem == 0)
        fn_mask = (sem == 0) & (gt_sem == 1)
        combined = sem.copy()
        combined[fp_mask] = 2   # FP -> red
        combined[fn_mask] = -2  # FN -> orange

        sem_colors_ext = {-1: "#DDDDDD", 0: "#95A5A6", 1: "#27AE60",
                          2: "#E74C3C",  # FP
                         -2: "#F39C12"}  # FN
        for val in [-1, 0, 1, 2, -2]:
            mask = combined == val
            if mask.sum() == 0:
                continue
            ax.scatter(x_all[mask], y_all[mask], s=0.08,
                       color=sem_colors_ext[val], alpha=0.7, rasterized=True)
        ax.set_title(title, fontweight="bold",
                     color=M_COLORS.get(method, "black"))
        ax.set_aspect("equal")
        ax.axis("off")
        fp_pct = 100 * fp_mask.sum() / len(sem)
        fn_pct = 100 * fn_mask.sum() / len(sem)
        ax.text(0.02, 0.02, f"FP={fp_pct:.2f}%  FN={fn_pct:.2f}%",
                transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Legend
    patches = [
        mpatches.Patch(color="#27AE60", label="Tree (correct)"),
        mpatches.Patch(color="#95A5A6", label="Non-tree (correct)"),
        mpatches.Patch(color="#E74C3C", label="False Positive"),
        mpatches.Patch(color="#F39C12", label="False Negative"),
    ]
    axes[2].legend(handles=patches, loc="upper right", fontsize=7,
                   framealpha=0.9, title="Classification")

    plt.tight_layout()
    plt.savefig(OUT / "fig_v3_semantic_overlay.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("[OK] fig_v3_semantic_overlay.png")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG-V4: Grid search landscape — RF+Density (cómo se calibró el segmentador)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_v4_gridsearch...")
df_gs = pd.read_csv("results/grid_search_rf_density.csv")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Watershed 3D Grid Search Landscape — RF + Density Seeding (val set)",
             fontsize=12, fontweight="bold")

params = [("voxel_size", "Voxel Size (m)"),
          ("gaussian_sigma", "Gaussian Sigma"),
          ("min_crown_radius_m", "Min Crown Radius (m)")]

for ax, (param, xlabel) in zip(axes, params):
    grouped = df_gs.groupby(param)["mean_f1"].max()
    bars = ax.bar(grouped.index.astype(str), grouped.values,
                  color="#2980B9", alpha=0.8, edgecolor="black", linewidth=1.2)
    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    ax.set_ylabel("Best F1 (val)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, grouped.max() * 1.2)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    # Highlight best
    best_val = grouped.idxmax()
    best_idx = list(grouped.index).index(best_val)
    bars[best_idx].set_color("#E74C3C")
    bars[best_idx].set_edgecolor("black")

plt.tight_layout()
plt.savefig(OUT / "fig_v4_gridsearch_landscape.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("[OK] fig_v4_gridsearch_landscape.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG-V5: Point cloud vertical profile (lateral view) — CULS
#         Muestra estructura 3D y colorea por instancia predicha vs GT
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_v5_vertical_profile...")
if HAS_LASPY:
    las = load_las("CULS", "plot_2_annotated")
    x_all = np.array(las.x)
    y_all = np.array(las.y)
    z_all = np.array(las.z)
    gt_ids = np.array(las["treeID"]).astype(int)

    # Corte transversal: franja de 5m en Y
    y_mid   = (y_all.min() + y_all.max()) / 2
    y_band  = 3.5
    mask_band = (y_all > y_mid - y_band) & (y_all < y_mid + y_band)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    fig.suptitle(
        "Vertical Cross-Section — CULS Plot 2  (7 m transect at plot center)",
        fontsize=12, fontweight="bold")

    # GT panel
    ax = axes[0]
    unique_gt = [g for g in np.unique(gt_ids) if g > 0]
    for i, gid in enumerate(unique_gt):
        mask = mask_band & (gt_ids == gid)
        if mask.sum() > 0:
            ax.scatter(x_all[mask], z_all[mask], s=0.3,
                       color=CMAP_INST(i % 25), rasterized=True)
    mask0 = mask_band & (gt_ids == 0)
    ax.scatter(x_all[mask0], z_all[mask0], s=0.1,
               color="#AAAAAA", alpha=0.3, rasterized=True)
    ax.set_title("Ground Truth", fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")

    # Best method
    for ax, method in zip(axes[1:], ["rf_density", "pointnet2"]):
        data = load_npz(method, "CULS__plot_2_annotated")
        if data is None:
            continue
        pred_ids = data["instance_ids"]
        unique_pred = [p for p in np.unique(pred_ids) if p > 0]
        sizes = [(pid, (pred_ids == pid).sum()) for pid in unique_pred]
        sizes.sort(key=lambda s: -s[1])
        for i, (pid, _) in enumerate(sizes):
            mask = mask_band & (pred_ids == pid)
            if mask.sum() > 0:
                ax.scatter(x_all[mask], z_all[mask], s=0.3,
                           color=CMAP_INST(i % 25), rasterized=True)
        mask0 = mask_band & (pred_ids == 0)
        ax.scatter(x_all[mask0], z_all[mask0], s=0.1,
                   color="#AAAAAA", alpha=0.3, rasterized=True)
        df = pd.read_csv(RESULTS)
        row = df[(df["method"] == method) &
                 (df["plot"].str.contains("CULS__plot_2"))]
        f1 = row["f1"].values[0] if len(row) else 0.0
        ax.set_title(f"{M_LABELS[method]}  (F1={f1:.3f})",
                     fontweight="bold", color=M_COLORS[method])
        ax.set_xlabel("X (m)")

    for ax in axes:
        ax.set_aspect("auto")
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(OUT / "fig_v5_vertical_profile.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("[OK] fig_v5_vertical_profile.png")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG-V6: Instance size distribution — predicted vs GT (RF+Density en CULS)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_v6_instance_size_dist...")
if HAS_LASPY:
    las = load_las("CULS", "plot_2_annotated")
    gt_ids = np.array(las["treeID"]).astype(int)
    gt_sizes = [(pred_ids == g).sum() for g in np.unique(gt_ids) if g > 0
                for pred_ids in [gt_ids]]

    data_best = load_npz("rf_density",  "CULS__plot_2_annotated")
    data_pn2  = load_npz("pointnet2",   "CULS__plot_2_annotated")
    data_base = load_npz("baseline",    "CULS__plot_2_annotated")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Predicted Instance Sizes vs Ground Truth — CULS Plot 2",
                 fontsize=12, fontweight="bold")

    gt_sizes_arr = np.array([
        (gt_ids == g).sum() for g in np.unique(gt_ids) if g > 0])

    def plot_size(ax, data, method, gt_arr):
        if data is None:
            return
        pred_ids = data["instance_ids"]
        pred_sizes = np.array([(pred_ids == p).sum()
                                for p in np.unique(pred_ids) if p > 0])
        bins = np.linspace(0, max(gt_arr.max(), pred_sizes.max()) * 1.05, 20)
        ax.hist(gt_arr, bins=bins, alpha=0.7, color="#27AE60",
                label=f"GT (n={len(gt_arr)})", edgecolor="white")
        ax.hist(pred_sizes, bins=bins, alpha=0.7,
                color=M_COLORS.get(method, "#2980B9"),
                label=f"Pred (n={len(pred_sizes)})", edgecolor="white")
        ax.axvline(np.median(gt_arr), color="#27AE60", lw=2,
                   linestyle="--", label=f"GT median={np.median(gt_arr):.0f}")
        ax.axvline(np.median(pred_sizes),
                   color=M_COLORS.get(method, "#2980B9"), lw=2,
                   linestyle="--", label=f"Pred median={np.median(pred_sizes):.0f}")
        ax.set_title(M_LABELS.get(method, method), fontweight="bold",
                     color=M_COLORS.get(method, "black"))
        ax.set_xlabel("Points per tree instance")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    plot_size(axes[0], data_base, "baseline",   gt_sizes_arr)
    plot_size(axes[1], data_best, "rf_density", gt_sizes_arr)
    plot_size(axes[2], data_pn2,  "pointnet2",  gt_sizes_arr)

    plt.tight_layout()
    plt.savefig(OUT / "fig_v6_instance_size_dist.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("[OK] fig_v6_instance_size_dist.png")
    plt.close()

print("""
[SUCCESS] Visual figures generated:
  fig_v1_topview_culs.png     -- top-view CULS: GT vs 4 methods
  fig_v2_topview_nibio.png    -- top-view NIBIO: GT vs 4 methods
  fig_v3_semantic_overlay.png -- semantic FP/FN overlay RF vs PN++
  fig_v4_gridsearch_landscape.png -- grid search landscape
  fig_v5_vertical_profile.png -- vertical cross-section 3D
  fig_v6_instance_size_dist.png  -- instance size distribution
""")
