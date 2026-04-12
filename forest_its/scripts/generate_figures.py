"""
generate_figures.py — All paper visualizations for the ITS comparison study.

Usage:
    python -m forest_its.scripts.generate_figures

Outputs:
    output/figures/01_dataset_overview/
    output/figures/02_semantic_results/
    output/figures/03_instance_results/
    output/figures/04_metrics/
    output/figures/paper_summary.html
"""

import sys
import base64
import traceback
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from forest_its.data.dataset import load_las, get_binary_labels
from forest_its.preprocessing.normalize_height import process_plot
from omegaconf import OmegaConf

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

METHOD_COLORS = {
    "Baseline": "#888888",
    "RF+WS":    "#2196F3",
    "PointNet2+WS":   "#FF5722",
}
METHOD_LABELS = ["Baseline", "RF+WS", "PointNet2+WS"]

SEM_COLORS = {
    0: "#CCCCCC",   # Unclassified
    1: "#90EE90",   # Low-veg
    2: "#8B4513",   # Terrain
    3: "#CCCCCC",   # Out-points
    4: "#8B0000",   # Stem
    5: "#006400",   # Live-branches
    6: "#DAA520",   # Woody-branches
}
SEM_LABELS = {
    1: "Low-veg", 2: "Terrain", 4: "Stem",
    5: "Live-branches", 6: "Woody-branches",
}

DATASET_ROOT = Path("C:/Users/cantu/Downloads/FORinstance_dataset")
OUTPUT_DIR   = Path("output")
FIGURES_DIR  = OUTPUT_DIR / "figures"

GENERATED: list[Path] = []
FAILED:    list[str]  = []

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("figures")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def save_fig(fig, path: Path, dpi: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    GENERATED.append(path)
    log.info(f"  Saved: {path}")
    return path


def load_plot_data(institution: str, plot_stem: str, cfg) -> dict:
    las_path = DATASET_ROOT / institution / f"{plot_stem}.las"
    data = load_las(las_path)
    data["_plot_stem"] = plot_stem
    process_plot(
        data,
        resolution_dtm=cfg.preprocessing.dtm_resolution,
        smooth_window=cfg.preprocessing.smooth_window,
        hag_min=cfg.preprocessing.hag_min,
        hag_max=cfg.preprocessing.hag_max,
    )
    return data


def load_predictions(method: str, plot_stem: str) -> dict:
    npz_path = OUTPUT_DIR / "predictions" / method / f"{plot_stem}_instances.npz"
    d = np.load(npz_path)
    return {k: d[k] for k in d}


def rasterize_xy(xyz, colors_rgb, resolution=0.05, bg=(1.0, 1.0, 1.0)):
    """Project points to XY plane, return RGBA image array."""
    x, y = xyz[:, 0], xyz[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    w = max(1, int((xmax - xmin) / resolution) + 1)
    h = max(1, int((ymax - ymin) / resolution) + 1)
    # cap at 4000px to avoid memory issues
    scale = 1.0
    if max(w, h) > 4000:
        scale = 4000 / max(w, h)
        w = int(w * scale)
        h = int(h * scale)
        resolution = (xmax - xmin) / w

    img = np.full((h, w, 3), bg, dtype=np.float32)
    xi = np.clip(((x - xmin) / resolution).astype(int), 0, w - 1)
    yi = np.clip(((y - ymin) / resolution).astype(int), 0, h - 1)
    # Reverse Y so north=up
    yi_img = h - 1 - yi
    img[yi_img, xi] = colors_rgb
    return img, xmin, xmax, ymin, ymax


def rasterize_xz(xyz, colors_rgb, resolution=0.05, y_center=None, y_band=5.0):
    """Project points in XZ plane (lateral view), within y_center +/- y_band."""
    if y_center is None:
        y_center = xyz[:, 1].mean()
    mask = np.abs(xyz[:, 1] - y_center) <= y_band
    if mask.sum() == 0:
        mask = np.ones(len(xyz), dtype=bool)
    xyz_s = xyz[mask]
    col_s = colors_rgb[mask]
    x, z = xyz_s[:, 0], xyz_s[:, 2]
    xmin, xmax = x.min(), x.max()
    zmin, zmax = z.min(), z.max()
    w = max(1, int((xmax - xmin) / resolution) + 1)
    h = max(1, int((zmax - zmin) / resolution) + 1)
    if max(w, h) > 3000:
        scale = 3000 / max(w, h)
        w = int(w * scale)
        h = int(h * scale)
        resolution_x = (xmax - xmin) / w
        resolution_z = (zmax - zmin) / h
    else:
        resolution_x = resolution
        resolution_z = resolution
    img = np.ones((h, w, 3), dtype=np.float32)
    xi = np.clip(((x - xmin) / resolution_x).astype(int), 0, w - 1)
    zi = np.clip(((z - zmin) / resolution_z).astype(int), 0, h - 1)
    zi_img = h - 1 - zi
    img[zi_img, xi] = col_s
    return img, xmin, xmax, zmin, zmax


def sem_class_to_rgb(classification):
    """Map raw Classification values to RGB float array."""
    rgb = np.ones((len(classification), 3), dtype=np.float32)
    for cls_val, hex_color in SEM_COLORS.items():
        mask = classification == cls_val
        r, g, b = mcolors.to_rgb(hex_color)
        rgb[mask] = [r, g, b]
    return rgb


def instance_id_to_rgb(instance_ids, seed=42):
    """Map instance_ids to distinct colors. 0 = grey background."""
    rng = np.random.RandomState(seed)
    unique_ids = np.unique(instance_ids)
    unique_ids = unique_ids[unique_ids != 0]
    n = len(unique_ids)
    if n == 0:
        return np.full((len(instance_ids), 3), 0.85, dtype=np.float32)
    if n <= 20:
        cmap = plt.cm.get_cmap("tab20", 20)
        palette = {uid: cmap(i % 20)[:3] for i, uid in enumerate(unique_ids)}
    else:
        hsv_vals = np.linspace(0, 1, n, endpoint=False)
        rng.shuffle(hsv_vals)
        palette = {}
        for i, uid in enumerate(unique_ids):
            h = hsv_vals[i]
            s = 0.8 + rng.uniform(0, 0.2)
            v = 0.7 + rng.uniform(0, 0.3)
            palette[uid] = mcolors.hsv_to_rgb([h, s, v])

    rgb = np.full((len(instance_ids), 3), 0.85, dtype=np.float32)
    for uid, color in palette.items():
        rgb[instance_ids == uid] = color
    return rgb


def binary_pred_to_rgb(pred_binary):
    """tree=dark green, no-tree=tan."""
    rgb = np.zeros((len(pred_binary), 3), dtype=np.float32)
    rgb[pred_binary == 0] = mcolors.to_rgb("#C8A97A")   # no-tree
    rgb[pred_binary == 1] = mcolors.to_rgb("#2D6A4F")   # tree
    return rgb


def add_scale_bar(ax, xmin, xmax, units="m", loc="lower right", fraction=0.2):
    """Add a simple scale bar to an image axes."""
    extent = xmax - xmin
    bar_m = round(extent * fraction / 10) * 10
    if bar_m < 1:
        bar_m = round(extent * fraction, 1)
    bar_frac = bar_m / extent
    x0, y0 = 0.75, 0.04
    ax.annotate("", xy=(x0 + bar_frac, y0), xytext=(x0, y0),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-", color="white", lw=2))
    ax.text(x0 + bar_frac / 2, y0 + 0.03, f"{bar_m:.0f} m",
            transform=ax.transAxes, ha="center", va="bottom",
            color="white", fontsize=7, fontweight="bold")


def fig_failed(name, exc):
    log.warning(f"  FAILED {name}: {exc}")
    FAILED.append(f"{name}: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# GROUP 1 — DATASET OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────

def fig_1_1_semantic_gt_overview(cfg):
    """3-col (CULS/NIBIO/SCION) x 2-row (top/lateral) semantic GT overview."""
    name = "1.1_semantic_gt_overview"
    try:
        plots_info = [
            ("CULS",  "plot_1_annotated", "CULS\n(open forest)"),
            ("NIBIO", "plot_2_annotated", "NIBIO\n(dense forest)"),
            ("SCION", "plot_35_annotated", "SCION\n(plantation)"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="white")
        fig.suptitle("Dataset Overview — Ground-Truth Semantic Labels", fontsize=14,
                     fontweight="bold", y=1.01)

        for col, (inst, stem, col_title) in enumerate(plots_info):
            log.info(f"    Loading {stem}...")
            data = load_plot_data(inst, stem, cfg)
            cls = data["classification"]
            xyz = data["xyz"]
            rgb = sem_class_to_rgb(cls)

            # Top view
            ax_top = axes[0, col]
            img_top, xmin, xmax, ymin, ymax = rasterize_xy(xyz, rgb)
            ax_top.imshow(img_top, aspect="equal",
                          extent=[xmin, xmax, ymin, ymax], origin="lower")
            n_pts = len(xyz)
            area_m2 = (xmax - xmin) * (ymax - ymin)
            density = n_pts / area_m2 if area_m2 > 0 else 0
            n_trees = int((data["tree_id"] > 0).sum() > 0)
            # Count unique tree IDs > 0
            n_trees_gt = len(np.unique(data["tree_id"][data["tree_id"] > 0]))
            ax_top.set_title(
                f"{col_title}\n{n_trees_gt} GT trees | {density:.0f} pts/m²",
                fontsize=9)
            ax_top.axis("off")
            add_scale_bar(ax_top, xmin, xmax)

            # Lateral view (XZ)
            ax_lat = axes[1, col]
            y_center = (ymin + ymax) / 2
            img_lat, xl, xr, zl, zr = rasterize_xz(xyz, rgb, y_center=y_center, y_band=5.0)
            ax_lat.imshow(img_lat, aspect="auto",
                          extent=[xl, xr, zl, zr], origin="lower")
            ax_lat.set_xlabel("X (m)", fontsize=7)
            if col == 0:
                ax_lat.set_ylabel("Z (m)", fontsize=7)
            ax_lat.tick_params(labelsize=6)

        axes[0, 0].set_ylabel("Top view (XY)", fontsize=8, labelpad=4)
        axes[1, 0].set_ylabel("Lateral view (XZ)\nY = center ± 5m", fontsize=8)

        # Semantic legend
        legend_patches = [
            mpatches.Patch(color=SEM_COLORS[k], label=v)
            for k, v in SEM_LABELS.items()
        ]
        legend_patches.append(mpatches.Patch(color="#CCCCCC", label="Excluded (0,3)"))
        fig.legend(handles=legend_patches, loc="lower center", ncol=6,
                   fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.03))

        fig.tight_layout()
        out = FIGURES_DIR / "01_dataset_overview" / "semantic_gt_overview.png"
        return save_fig(fig, out)
    except Exception as e:
        fig_failed(name, e)
        traceback.print_exc()


def fig_1_2_dataset_stats(cfg):
    """2x2 panel: density, GT trees, class distribution, density vs F1."""
    name = "1.2_dataset_stats"
    try:
        # Load val metrics
        rf_df   = pd.read_csv(OUTPUT_DIR / "results" / "rf_metrics_val.csv")
        base_df = pd.read_csv(OUTPUT_DIR / "results" / "baseline_metrics_val.csv")
        pointnet2_df  = pd.read_csv(OUTPUT_DIR / "results" / "pointnet2_metrics_val.csv")

        # Compute per-institution means (val plots only)
        insts = ["CULS", "NIBIO", "SCION"]
        inst_colors = {"CULS": "#4CAF50", "NIBIO": "#2196F3", "SCION": "#FF9800"}

        # Point density and GT trees from val plots
        density_by_inst = {}
        trees_by_inst = {}
        for inst in insts:
            sub = rf_df[rf_df["institution"] == inst]
            trees_by_inst[inst] = sub["n_gt_trees"].sum()
            # Density: pts evaluated / plot area (approximate from sem_n_points)
            # Use a rough area estimate from # points at 200 pts/m2 typical
            density_by_inst[inst] = sub["sem_n_points_evaluated"].mean() / 30000

        fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor="white")
        fig.suptitle("Dataset Statistics (Val Set, CULS / NIBIO / SCION)", fontsize=13,
                     fontweight="bold")

        # a) Point density by institution
        ax = axes[0, 0]
        bar_colors = [inst_colors[i] for i in insts]
        vals = [density_by_inst[i] for i in insts]
        bars = ax.bar(insts, vals, color=bar_colors, edgecolor="white", width=0.5)
        ax.set_ylabel("Approx. pts/m² (×1000)", fontsize=9)
        ax.set_title("a) Mean Point Density by Institution", fontsize=10)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.0f}k", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)

        # b) GT trees by institution
        ax = axes[0, 1]
        vals_t = [trees_by_inst[i] for i in insts]
        bars = ax.bar(insts, vals_t, color=bar_colors, edgecolor="white", width=0.5)
        ax.set_ylabel("# GT trees (val plots)", fontsize=9)
        ax.set_title("b) Ground-Truth Trees by Institution", fontsize=10)
        for bar, v in zip(bars, vals_t):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    str(v), ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(vals_t) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)

        # c) Semantic class distribution (from RF metrics, using mIoU proxy)
        # Use sem_iou_tree vs sem_iou_notree as proxy for class difficulty
        ax = axes[1, 0]
        iou_tree    = [rf_df[rf_df["institution"]==i]["sem_iou_tree"].mean() for i in insts]
        iou_notree  = [rf_df[rf_df["institution"]==i]["sem_iou_notree"].mean() for i in insts]
        x = np.arange(len(insts))
        w = 0.35
        ax.bar(x - w/2, iou_tree,   w, label="IoU Tree",    color="#2D6A4F", alpha=0.85)
        ax.bar(x + w/2, iou_notree, w, label="IoU No-tree", color="#C8A97A", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(insts)
        ax.set_ylabel("Semantic IoU (RF)", fontsize=9)
        ax.set_title("c) Semantic IoU by Class and Institution (RF)", fontsize=10)
        ax.set_ylim(0.9, 1.01)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        # d) Density vs F1 (RF) scatter — the key correlation
        ax = axes[1, 1]
        for inst, color in inst_colors.items():
            sub = rf_df[rf_df["institution"] == inst]
            dens = sub["sem_n_points_evaluated"] / 30000
            ax.scatter(dens, sub["f1"], color=color, s=80, label=inst,
                       zorder=3, edgecolors="white", linewidths=0.5)
            for _, row in sub.iterrows():
                ax.annotate(row["plot"].replace("_annotated",""),
                            (row["sem_n_points_evaluated"]/30000, row["f1"]),
                            fontsize=6, ha="left", va="bottom",
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Approx. point density (×1000 pts/m²)", fontsize=9)
        ax.set_ylabel("Instance F1 (RF+WS)", fontsize=9)
        ax.set_title("d) Point Density vs. Instance F1 (RF)", fontsize=10)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        out = FIGURES_DIR / "01_dataset_overview" / "dataset_stats.png"
        return save_fig(fig, out)
    except Exception as e:
        fig_failed(name, e)
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# GROUP 2 — SEMANTIC RESULTS
# ──────────────────────────────────────────────────────────────────────────────

def _sem_comparison_one_plot(data, pred_rf, pred_pointnet2, stem, rf_row, pointnet2_row):
    """Create the 3-col semantic comparison figure for one plot."""
    cls = data["classification"]
    xyz = data["xyz"]
    rgb_gt = sem_class_to_rgb(cls)

    pred_rf_bin  = pred_rf.get("semantic_pred", np.zeros(len(xyz), dtype=np.int32))
    pred_pointnet2_bin = pred_pointnet2.get("semantic_pred", np.zeros(len(xyz), dtype=np.int32))
    rgb_rf  = binary_pred_to_rgb(pred_rf_bin)
    rgb_pointnet2 = binary_pred_to_rgb(pred_pointnet2_bin)

    rf_miou  = rf_row["sem_miou"]  if rf_row is not None else float("nan")
    pointnet2_miou = pointnet2_row["sem_miou"] if pointnet2_row is not None else float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    titles = [
        "Ground-Truth Semantic Labels",
        f"RF Prediction\nmIoU = {rf_miou:.4f}",
        f"PointNet2 Prediction\nmIoU = {pointnet2_miou:.4f}",
    ]
    rgbs = [rgb_gt, rgb_rf, rgb_pointnet2]

    for ax, rgb, title in zip(axes, rgbs, titles):
        img, xmin, xmax, ymin, ymax = rasterize_xy(xyz, rgb)
        ax.imshow(img, aspect="equal",
                  extent=[xmin, xmax, ymin, ymax], origin="lower")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        add_scale_bar(ax, xmin, xmax)

    # Legends
    gt_patches = [mpatches.Patch(color=SEM_COLORS[k], label=v)
                  for k, v in SEM_LABELS.items()]
    pred_patches = [
        mpatches.Patch(color="#2D6A4F", label="Tree (pred)"),
        mpatches.Patch(color="#C8A97A", label="No-tree (pred)"),
    ]
    axes[0].legend(handles=gt_patches, loc="lower left", fontsize=6,
                   framealpha=0.7, ncol=2)
    for ax in axes[1:]:
        ax.legend(handles=pred_patches, loc="lower left", fontsize=7,
                  framealpha=0.7)

    inst = stem.split("_")[0] if "_" in stem else stem
    fig.suptitle(f"Semantic Comparison — {stem.replace('_annotated', '')}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_2_1_semantic_comparison(cfg):
    """Semantic GT vs RF vs PointNet2 for CULS plot_1 and NIBIO plot_2."""
    name = "2.1_semantic_comparison"
    try:
        rf_df  = pd.read_csv(OUTPUT_DIR / "results" / "rf_metrics_val.csv")
        pointnet2_df = pd.read_csv(OUTPUT_DIR / "results" / "pointnet2_metrics_val.csv")

        target_plots = [
            ("CULS",  "plot_1_annotated"),
            ("NIBIO", "plot_2_annotated"),
        ]
        for inst, stem in target_plots:
            log.info(f"    Semantic comparison: {stem}")
            data     = load_plot_data(inst, stem, cfg)
            pred_rf  = load_predictions("rf",  stem)
            pred_pointnet2 = load_predictions("pointnet2", stem)

            rf_row  = rf_df[rf_df["plot"]  == stem].iloc[0] if len(rf_df[rf_df["plot"]  == stem]) else None
            pointnet2_row = pointnet2_df[pointnet2_df["plot"] == stem].iloc[0] if len(pointnet2_df[pointnet2_df["plot"] == stem]) else None

            fig = _sem_comparison_one_plot(data, pred_rf, pred_pointnet2, stem, rf_row, pointnet2_row)
            out = FIGURES_DIR / "02_semantic_results" / f"semantic_comparison_{stem.replace('_annotated','')}.png"
            save_fig(fig, out)
    except Exception as e:
        fig_failed(name, e)
        traceback.print_exc()


def fig_2_2_semantic_error_maps(cfg):
    """Error maps (TP/FP/FN/TN) for RF and PointNet2 on CULS plot_1 and NIBIO plot_2."""
    name = "2.2_semantic_error_maps"
    ERR_COLORS = {
        "TP":  "#2D6A4F",   # green
        "TN":  "#5D8AA8",   # blue
        "FP":  "#C0392B",   # red
        "FN":  "#E67E22",   # orange
        "Excl": "#CCCCCC",  # grey
    }
    try:
        pointnet2_df = pd.read_csv(OUTPUT_DIR / "results" / "pointnet2_metrics_val.csv")
        rf_df  = pd.read_csv(OUTPUT_DIR / "results" / "rf_metrics_val.csv")

        target_plots = [
            ("CULS",  "plot_1_annotated"),
            ("NIBIO", "plot_2_annotated"),
        ]
        for inst, stem in target_plots:
            log.info(f"    Error map: {stem}")
            data     = load_plot_data(inst, stem, cfg)
            pred_rf  = load_predictions("rf",  stem)
            pred_pointnet2 = load_predictions("pointnet2", stem)
            xyz  = data["xyz"]
            cls  = data["classification"]
            gt   = get_binary_labels(cls)   # -1, 0, 1

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
            for ax, (pred_bin, method_name) in zip(axes, [
                (pred_rf.get("semantic_pred"), "RF"),
                (pred_pointnet2.get("semantic_pred"), "PointNet2"),
            ]):
                if pred_bin is None:
                    ax.set_title(f"{method_name} — no pred"); continue
                rgb = np.zeros((len(xyz), 3), dtype=np.float32)
                excl  = gt == -1
                tp    = (~excl) & (pred_bin == 1) & (gt == 1)
                tn    = (~excl) & (pred_bin == 0) & (gt == 0)
                fp    = (~excl) & (pred_bin == 1) & (gt == 0)
                fn    = (~excl) & (pred_bin == 0) & (gt == 1)
                for key, mask in [("Excl", excl), ("TN", tn), ("TP", tp), ("FP", fp), ("FN", fn)]:
                    rgb[mask] = mcolors.to_rgb(ERR_COLORS[key])

                img, xmin, xmax, ymin, ymax = rasterize_xy(xyz, rgb)
                ax.imshow(img, aspect="equal",
                          extent=[xmin, xmax, ymin, ymax], origin="lower")
                ax.axis("off")
                add_scale_bar(ax, xmin, xmax)
                err_pct = 100 * (fp.sum() + fn.sum()) / max(1, (~excl).sum())
                ax.set_title(f"{method_name} Error Map — {stem.replace('_annotated','')}\n"
                             f"Error rate: {err_pct:.2f}%", fontsize=9)

            legend_patches = [mpatches.Patch(color=c, label=k)
                              for k, c in ERR_COLORS.items()]
            fig.legend(handles=legend_patches, loc="lower center", ncol=5,
                       fontsize=8, bbox_to_anchor=(0.5, -0.05))
            fig.tight_layout()
            out = FIGURES_DIR / "02_semantic_results" / \
                  f"semantic_error_map_{stem.replace('_annotated','')}.png"
            save_fig(fig, out)
    except Exception as e:
        fig_failed(name, e)
        traceback.print_exc()


def fig_2_3_confusion_matrices(cfg):
    """2x2 confusion matrices: RF and PointNet2 on CULS and NIBIO."""
    name = "2.3_confusion_matrices"
    try:
        target = [
            ("CULS",  "plot_1_annotated"),
            ("NIBIO", "plot_2_annotated"),
        ]
        methods = [("rf", "RF"), ("pointnet2", "PointNet2")]

        fig, axes = plt.subplots(2, 2, figsize=(10, 9), facecolor="white")
        fig.suptitle("Semantic Confusion Matrices (Recall-normalized)", fontsize=12,
                     fontweight="bold")

        for row_idx, (inst, stem) in enumerate(target):
            data    = load_plot_data(inst, stem, cfg)
            gt      = get_binary_labels(data["classification"])
            valid   = gt != -1
            gt_v    = gt[valid]

            for col_idx, (method_key, method_label) in enumerate(methods):
                pred    = load_predictions(method_key, stem)
                pred_v  = pred.get("semantic_pred", np.zeros(len(gt), dtype=np.int32))[valid]
                classes = ["No-tree", "Tree"]
                n = len(classes)
                cm = np.zeros((n, n), dtype=float)
                for i in range(n):
                    for j in range(n):
                        cm[i, j] = ((gt_v == i) & (pred_v == j)).sum()
                cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(1)

                ax = axes[row_idx, col_idx]
                im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
                ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
                ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
                ax.set_xlabel("Predicted", fontsize=8)
                ax.set_ylabel("Ground-Truth", fontsize=8)
                ax.set_title(f"{method_label} — {inst}\n{stem.replace('_annotated','')}", fontsize=9)
                for i in range(n):
                    for j in range(n):
                        ax.text(j, i, f"{int(cm[i,j]):,}\n({cm_norm[i,j]:.2f})",
                                ha="center", va="center", fontsize=8,
                                color="white" if cm_norm[i,j] > 0.6 else "black")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        out = FIGURES_DIR / "02_semantic_results" / "confusion_matrices.png"
        return save_fig(fig, out)
    except Exception as e:
        fig_failed(name, e)
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# GROUP 3 — INSTANCE RESULTS
# ──────────────────────────────────────────────────────────────────────────────

def _instance_panel(data, stem, metrics_by_method: dict, out_path: Path, title: str):
    """4-col instance comparison panel: GT / Baseline / RF / PointNet2."""
    xyz = data["xyz"]
    gt_ids = data["tree_id"]
    rgb_gt = instance_id_to_rgb(gt_ids)

    methods_info = [
        ("baseline", "Baseline"),
        ("rf",       "RF+WS"),
        ("pointnet2", "PointNet2+WS"),
    ]
    preds = []
    for mkey, mlabel in methods_info:
        try:
            p = load_predictions(mkey, stem)
            preds.append((mlabel, p["instance_ids"]))
        except Exception as ex:
            log.warning(f"      Could not load {mkey} predictions: {ex}")
            preds.append((mlabel, np.zeros(len(xyz), dtype=np.int32)))

    n_gt = len(np.unique(gt_ids[gt_ids > 0]))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), facecolor="white")
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # GT
    img_gt, xmin, xmax, ymin, ymax = rasterize_xy(xyz, rgb_gt)
    axes[0].imshow(img_gt, aspect="equal",
                   extent=[xmin, xmax, ymin, ymax], origin="lower")
    axes[0].set_title(f"Ground-Truth Instance\n({n_gt} trees)", fontsize=9)
    axes[0].axis("off")
    add_scale_bar(axes[0], xmin, xmax)

    for ax, (mlabel, inst_ids) in zip(axes[1:], preds):
        rgb_pred = instance_id_to_rgb(inst_ids)
        img, *_ = rasterize_xy(xyz, rgb_pred)
        ax.imshow(img, aspect="equal",
                  extent=[xmin, xmax, ymin, ymax], origin="lower")
        n_pred = len(np.unique(inst_ids[inst_ids > 0]))
        m = metrics_by_method.get(mlabel, {})
        f1   = m.get("f1", float("nan"))
        prec = m.get("precision", float("nan"))
        rec  = m.get("recall", float("nan"))
        ax.set_title(
            f"{mlabel}\n{n_pred} pred / {n_gt} GT | F1={f1:.3f}\n"
            f"Prec={prec:.3f}  Rec={rec:.3f}",
            fontsize=8)
        ax.axis("off")
        add_scale_bar(ax, xmin, xmax)

    fig.tight_layout()
    return save_fig(fig, out_path)


def fig_3_1_instance_culs_plot1(cfg):
    name = "3.1_instance_culs_plot1"
    try:
        stem = "plot_1_annotated"
        inst = "CULS"
        data = load_plot_data(inst, stem, cfg)

        rf_df   = pd.read_csv(OUTPUT_DIR / "results" / "rf_metrics_val.csv")
        base_df = pd.read_csv(OUTPUT_DIR / "results" / "baseline_metrics_val.csv")
        pointnet2_df  = pd.read_csv(OUTPUT_DIR / "results" / "pointnet2_metrics_val.csv")

        def get_m(df, s): return df[df["plot"] == s].iloc[0].to_dict() if len(df[df["plot"]==s]) else {}
        metrics = {
            "Baseline": get_m(base_df, stem),
            "RF+WS":    get_m(rf_df, stem),
            "PointNet2+WS":   get_m(pointnet2_df, stem),
        }
        out = FIGURES_DIR / "03_instance_results" / "instance_comparison_culs_plot1.png"
        _instance_panel(data, stem, metrics, out,
                        f"Instance Segmentation — CULS plot_1 (open forest, 6 GT trees)")
    except Exception as e:
        fig_failed(name, e)
        traceback.print_exc()


def fig_3_2_instance_nibio_plot2(cfg):
    name = "3.2_instance_nibio_plot2"
    try:
        stem = "plot_2_annotated"
        inst = "NIBIO"
        data = load_plot_data(inst, stem, cfg)

        rf_df   = pd.read_csv(OUTPUT_DIR / "results" / "rf_metrics_val.csv")
        base_df = pd.read_csv(OUTPUT_DIR / "results" / "baseline_metrics_val.csv")
        pointnet2_df  = pd.read_csv(OUTPUT_DIR / "results" / "pointnet2_metrics_val.csv")

        def get_m(df, s): return df[df["plot"] == s].iloc[0].to_dict() if len(df[df["plot"]==s]) else {}
        metrics = {
            "Baseline": get_m(base_df, stem),
            "RF+WS":    get_m(rf_df, stem),
            "PointNet2+WS":   get_m(pointnet2_df, stem),
        }
        out = FIGURES_DIR / "03_instance_results" / "instance_comparison_nibio_plot2.png"
        _instance_panel(data, stem, metrics, out,
                        "Instance Segmentation — NIBIO plot_2 (dense forest, 40 GT trees) — "
                        "Watershed bottleneck")
    except Exception as e:
        fig_failed(name, e)
        traceback.print_exc()


def fig_3_3_instance_3d_lateral(cfg):
    """3D lateral view via matplotlib XZ projection (open3d fallback)."""
    name = "3.3_instance_3d_lateral"
    try:
        stem = "plot_1_annotated"
        data = load_plot_data("CULS", stem, cfg)
        xyz  = data["xyz"]
        gt_ids = data["tree_id"]

        pred_rf  = load_predictions("rf", stem)
        pred_pointnet2 = load_predictions("pointnet2", stem)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
        fig.suptitle("CULS plot_1 — Lateral View (XZ projection)\n"
                     "Instance Segmentation Comparison", fontsize=12, fontweight="bold")

        datasets = [
            ("Ground-Truth",    gt_ids),
            ("RF+WS Predicted", pred_rf["instance_ids"]),
            ("PointNet2+WS Predicted", pred_pointnet2["instance_ids"]),
        ]
        for ax, (title, ids) in zip(axes, datasets):
            rgb = instance_id_to_rgb(ids)
            y_center = xyz[:, 1].mean()
            img, xl, xr, zl, zr = rasterize_xz(xyz, rgb, y_center=y_center, y_band=8.0,
                                                  resolution=0.1)
            ax.imshow(img, aspect="auto",
                      extent=[xl, xr, zl, zr], origin="lower")
            n = len(np.unique(ids[ids > 0]))
            ax.set_title(f"{title}\n({n} trees)", fontsize=9)
            ax.set_xlabel("X (m)", fontsize=8)
            ax.set_ylabel("Z / Height (m)", fontsize=8)
            ax.tick_params(labelsize=7)

        fig.tight_layout()
        out = FIGURES_DIR / "03_instance_results" / "instance_3d_lateral_culs_plot1.png"
        return save_fig(fig, out)
    except Exception as e:
        fig_failed(name, e)
        traceback.print_exc()


def fig_3_4_underseg_illustration(cfg):
    """Zoom on a 20x20m area of NIBIO plot_2 showing under-segmentation."""
    name = "3.4_underseg_illustration"
    try:
        stem = "plot_2_annotated"
        data = load_plot_data("NIBIO", stem, cfg)
        xyz  = data["xyz"]
        gt_ids = data["tree_id"]
        pred_rf = load_predictions("rf", stem)
        pred_ids = pred_rf["instance_ids"]

        # Find a 20x20m zone with most GT trees
        xc = xyz[:, 0].mean()
        yc = xyz[:, 1].mean()

        best_zone = (xc, yc)
        best_n = 0
        for dx in np.linspace(-50, 50, 11):
            for dy in np.linspace(-50, 50, 11):
                cx, cy = xc + dx, yc + dy
                mask = ((np.abs(xyz[:, 0] - cx) < 10) &
                        (np.abs(xyz[:, 1] - cy) < 10) &
                        (gt_ids > 0))
                n_trees = len(np.unique(gt_ids[mask]))
                if n_trees > best_n:
                    best_n = n_trees
                    best_zone = (cx, cy)

        zx, zy = best_zone
        zone_mask = ((np.abs(xyz[:, 0] - zx) < 10) &
                     (np.abs(xyz[:, 1] - zy) < 10))

        xyz_z   = xyz[zone_mask]
        gt_z    = gt_ids[zone_mask]
        pred_z  = pred_ids[zone_mask]

        rgb_gt   = instance_id_to_rgb(gt_z)
        rgb_pred = instance_id_to_rgb(pred_z)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="white")
        fig.suptitle("Under-Segmentation Illustration — NIBIO plot_2\n"
                     "20×20m zoom region", fontsize=11, fontweight="bold")

        for ax, rgb, title, ids in [
            (axes[0], rgb_gt,   "Ground-Truth", gt_z),
            (axes[1], rgb_pred, "RF+WS Prediction", pred_z),
        ]:
            img, xmin, xmax, ymin, ymax = rasterize_xy(xyz_z, rgb, resolution=0.02)
            ax.imshow(img, aspect="equal",
                      extent=[xmin, xmax, ymin, ymax], origin="lower")
            n = len(np.unique(ids[ids > 0]))
            ax.set_title(f"{title}\n({n} {'trees' if n != 1 else 'tree'})", fontsize=10)
            ax.set_xlabel("X (m)", fontsize=8)
            ax.set_ylabel("Y (m)", fontsize=8)
            ax.tick_params(labelsize=7)

            # Circle merged trees in prediction
            if title != "Ground-Truth":
                gt_tree_ids = np.unique(gt_z[gt_z > 0])
                for pid in np.unique(ids[ids > 0]):
                    pts = xyz_z[ids == pid]
                    cx_p, cy_p = pts[:, 0].mean(), pts[:, 1].mean()
                    gt_here = len(np.unique(gt_z[ids == pid]))
                    if gt_here >= 3:
                        circle = plt.Circle((cx_p, cy_p), 2.5,
                                            fill=False, color="white",
                                            linestyle="--", linewidth=1.5)
                        ax.add_patch(circle)
                        ax.text(cx_p, cy_p + 3, f"{gt_here} GT\n1 pred",
                                ha="center", va="bottom", color="white",
                                fontsize=6, fontweight="bold")

        fig.tight_layout()
        out = FIGURES_DIR / "03_instance_results" / "underseg_illustration.png"
        return save_fig(fig, out)
    except Exception as e:
        fig_failed(name, e)
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# GROUP 4 — METRICS
# ──────────────────────────────────────────────────────────────────────────────

def fig_4_1_f1_by_institution():
    """Grouped bar chart: F1 by institution and method."""
    name = "4.1_f1_by_institution"
    try:
        rf_df   = pd.read_csv(OUTPUT_DIR / "results" / "rf_metrics_val.csv")
        base_df = pd.read_csv(OUTPUT_DIR / "results" / "baseline_metrics_val.csv")
        pointnet2_df  = pd.read_csv(OUTPUT_DIR / "results" / "pointnet2_metrics_val.csv")

        insts = ["CULS", "NIBIO", "SCION"]
        data_by_method = {
            "Baseline": base_df,
            "RF+WS":    rf_df,
            "PointNet2+WS":   pointnet2_df,
        }

        means = {}
        stds  = {}
        for mname, df in data_by_method.items():
            means[mname] = []
            stds[mname]  = []
            for inst in insts:
                vals = df[df["institution"] == inst]["f1"].values
                means[mname].append(vals.mean() if len(vals) else 0.0)
                stds[mname].append(vals.std() if len(vals) > 1 else 0.0)

        x = np.arange(len(insts))
        width = 0.22

        fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
        for i, (mname, color) in enumerate(METHOD_COLORS.items()):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, means[mname], width,
                          label=mname, color=color, alpha=0.90,
                          yerr=stds[mname], capsize=4, error_kw={"elinewidth": 1.2})
            for bar, v in zip(bars, means[mname]):
                if v > 0.01:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + (stds[mname][list(means[mname]).index(v)] + 0.02),
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                            fontweight="bold")

        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.6,
                   label="F1 = 0.5 reference")
        ax.set_xticks(x)
        ax.set_xticklabels(insts, fontsize=11)
        ax.set_ylabel("Instance F1", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_title("Instance Segmentation F1 by Institution and Method\n"
                     "(Val Set, IoU threshold = 0.5)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        # Annotation
        ax.text(2, 0.05, "Watershed\nbottleneck\n(dense forest)",
                ha="center", va="bottom", fontsize=8, color="#888",
                style="italic")

        fig.tight_layout()
        out = FIGURES_DIR / "04_metrics" / "f1_by_institution.png"
        return save_fig(fig, out)
    except Exception as e:
        fig_failed(name, e)
        traceback.print_exc()


def fig_4_2_decoupling_scatter():
    """sem_mIoU vs F1 instance — the decoupling figure."""
    name = "4.2_semantic_vs_instance_decoupling"
    try:
        rf_df  = pd.read_csv(OUTPUT_DIR / "results" / "rf_metrics_val.csv")
        pointnet2_df = pd.read_csv(OUTPUT_DIR / "results" / "pointnet2_metrics_val.csv")

        INST_MARKERS = {"CULS": "o", "NIBIO": "s", "SCION": "^"}
        INST_LABELS  = {"CULS": "CULS (open)", "NIBIO": "NIBIO (dense)", "SCION": "SCION (plantation)"}

        fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")

        # Shaded decoupling zone: sem_mIoU > 0.95, F1 < 0.2
        ax.axvspan(0.95, 1.01, 0, 0.2, color="#FFEBEE", alpha=0.6,
                   label="Decoupling zone\n(high sem. quality, low instance F1)")
        ax.annotate("Decoupling\nzone", (0.975, 0.08), fontsize=8,
                    color="#C0392B", ha="center", style="italic")

        plotted_methods = set()
        plotted_insts   = set()
        for df, method_label, color in [
            (rf_df,  "RF+WS",  METHOD_COLORS["RF+WS"]),
            (pointnet2_df, "PointNet2+WS", METHOD_COLORS["PointNet2+WS"]),
        ]:
            for inst, marker in INST_MARKERS.items():
                sub = df[df["institution"] == inst]
                for _, row in sub.iterrows():
                    if "sem_miou" not in row or pd.isna(row["sem_miou"]):
                        continue
                    n_trees = row["n_gt_trees"]
                    size = 60 + n_trees * 3
                    label_m = method_label if method_label not in plotted_methods else None
                    label_i = INST_LABELS[inst] if inst not in plotted_insts else None
                    ax.scatter(row["sem_miou"], row["f1"],
                               color=color, marker=marker, s=size,
                               edgecolors="white", linewidths=1.5, zorder=4,
                               alpha=0.9)
                    # Annotate with plot name + method
                    ax.annotate(
                        f"{row['plot'].replace('_annotated','')} ({method_label})",
                        (row["sem_miou"], row["f1"]),
                        fontsize=6.5, ha="left", va="bottom",
                        xytext=(5, 3), textcoords="offset points", color=color)
                    plotted_methods.add(method_label)
                    plotted_insts.add(inst)

        # Custom legend
        method_handles = [
            mpatches.Patch(color=METHOD_COLORS["RF+WS"],  label="RF+WS"),
            mpatches.Patch(color=METHOD_COLORS["PointNet2+WS"], label="PointNet2+WS"),
        ]
        inst_handles = [
            plt.Line2D([0], [0], marker="o", color="grey", linestyle="none", ms=8, label="CULS (open)"),
            plt.Line2D([0], [0], marker="s", color="grey", linestyle="none", ms=8, label="NIBIO (dense)"),
            plt.Line2D([0], [0], marker="^", color="grey", linestyle="none", ms=8, label="SCION (plantation)"),
        ]
        l1 = ax.legend(handles=method_handles, loc="upper left", fontsize=9, title="Method")
        ax.add_artist(l1)
        ax.legend(handles=inst_handles, loc="center left", fontsize=9, title="Institution")

        ax.set_xlabel("Semantic mIoU (full plot)", fontsize=11)
        ax.set_ylabel("Instance F1 (IoU threshold = 0.5)", fontsize=11)
        ax.set_title("Semantic Quality vs. Instance Quality\n"
                     "— Decoupling: improving semantics does not improve instance segmentation —",
                     fontsize=11, fontweight="bold")
        ax.set_xlim(0.97, 1.002)
        ax.set_ylim(-0.02, 1.05)
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        out = FIGURES_DIR / "04_metrics" / "semantic_vs_instance_decoupling.png"
        return save_fig(fig, out)
    except Exception as e:
        fig_failed(name, e)
        traceback.print_exc()


def fig_4_3_underseg_overseg():
    """Horizontal stacked diverging bar chart: under vs over segmentation."""
    name = "4.3_underseg_overseg"
    try:
        rf_df   = pd.read_csv(OUTPUT_DIR / "results" / "rf_metrics_val.csv")
        base_df = pd.read_csv(OUTPUT_DIR / "results" / "baseline_metrics_val.csv")
        pointnet2_df  = pd.read_csv(OUTPUT_DIR / "results" / "pointnet2_metrics_val.csv")

        plots_order = ["plot_1_annotated", "plot_3_annotated",
                       "plot_9_annotated", "plot_2_annotated", "plot_35_annotated"]
        methods_data = [
            ("Baseline", base_df, METHOD_COLORS["Baseline"]),
            ("RF+WS",    rf_df,   METHOD_COLORS["RF+WS"]),
            ("PointNet2+WS",   pointnet2_df,  METHOD_COLORS["PointNet2+WS"]),
        ]

        rows = []
        for stem in plots_order:
            for mname, df, color in methods_data:
                sub = df[df["plot"] == stem]
                if len(sub) == 0:
                    continue
                r = sub.iloc[0]
                inst = r["institution"]
                short = stem.replace("_annotated", "")
                rows.append({
                    "label":    f"{short}\n({inst})",
                    "method":   mname,
                    "under":    r.get("under_seg", 0),
                    "over":     r.get("over_seg", 0),
                    "color":    color,
                })

        fig, ax = plt.subplots(figsize=(10, 9), facecolor="white")
        yticks, ylabels = [], []
        y = 0
        gap_between_plots = 0.3
        bar_h = 0.22

        seen_labels = set()
        prev_stem = None
        for row in rows:
            stem = row["label"].split("\n")[0]
            if prev_stem != stem:
                if prev_stem is not None:
                    y += gap_between_plots
                prev_stem = stem

            # Under-seg bar (left)
            ax.barh(y, -row["under"], height=bar_h, color=row["color"], alpha=0.85,
                    label=row["method"] if row["method"] not in seen_labels else None)
            seen_labels.add(row["method"])
            # Over-seg bar (right)
            ax.barh(y, row["over"], height=bar_h, color=row["color"], alpha=0.85)
            ax.text(-row["under"] - 0.01, y, f"-{row['under']:.2f}",
                    ha="right", va="center", fontsize=6.5)
            ax.text(row["over"] + 0.01,  y, f"{row['over']:.2f}",
                    ha="left",  va="center", fontsize=6.5)

            yticks.append(y)
            ylabels.append(f"{row['label']}\n{row['method']}")
            y -= bar_h + 0.06

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=6.5)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_xlabel("← Under-segmentation          Over-segmentation →", fontsize=9)
        ax.set_title("Under- vs. Over-Segmentation by Plot and Method\n"
                     "(Val Set)", fontsize=11, fontweight="bold")
        ax.set_xlim(-1.1, 0.65)
        ax.legend(fontsize=9, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        out = FIGURES_DIR / "04_metrics" / "underseg_overseg_comparison.png"
        return save_fig(fig, out)
    except Exception as e:
        fig_failed(name, e)
        traceback.print_exc()


def fig_4_4_training_curve():
    """PointNet2 training curve: train/val mIoU by epoch."""
    name = "4.4_training_curve"
    try:
        log_path = OUTPUT_DIR / "logs" / "train_pointnet2.log"
        if not log_path.exists():
            fig_failed(name, f"log not found: {log_path}")
            return

        epochs, train_miou, val_miou, train_loss, val_loss = [], [], [], [], []
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "Epoch" not in line or "/100" not in line:
                    continue
                import re
                m = re.search(
                    r"Epoch\s+(\d+)/100.*Train Loss:\s*([\d.]+)\s+mIoU:\s*([\d.]+)"
                    r".*Val Loss:\s*([\d.]+)\s+mIoU:\s*([\d.]+)", line)
                if m:
                    epochs.append(int(m.group(1)))
                    train_loss.append(float(m.group(2)))
                    train_miou.append(float(m.group(3)))
                    val_loss.append(float(m.group(4)))
                    val_miou.append(float(m.group(5)))

        if not epochs:
            fig_failed(name, "no epoch data parsed")
            return

        epochs = np.array(epochs)
        best_ep = epochs[np.argmax(val_miou)]
        best_miou = max(val_miou)

        fig, ax1 = plt.subplots(figsize=(10, 5.5), facecolor="white")
        ax2 = ax1.twinx()

        ax1.plot(epochs, train_loss, color="#888", linewidth=1.2, alpha=0.7, label="Train Loss")
        ax1.plot(epochs, val_loss,   color="#555", linewidth=1.2, alpha=0.7,
                 linestyle="--", label="Val Loss")
        ax1.set_ylabel("Loss (NLL)", fontsize=10, color="#555")
        ax1.set_ylim(0, max(max(train_loss), max(val_loss)) * 1.15)
        ax1.tick_params(axis="y", labelcolor="#555")

        ax2.plot(epochs, train_miou, color="#2196F3", linewidth=2.0, label="Train mIoU")
        ax2.plot(epochs, val_miou,   color="#FF5722", linewidth=2.0, label="Val mIoU")
        ax2.axhspan(0.80, 1.01, color="#E8F5E9", alpha=0.4, label="mIoU > 0.80 zone")
        ax2.axvline(best_ep, color="#FF5722", linewidth=1.5, linestyle=":",
                    label=f"Best epoch {best_ep} (mIoU={best_miou:.4f})")
        ax2.scatter([best_ep], [best_miou], color="#FF5722", s=80, zorder=5)
        ax2.set_ylabel("mIoU", fontsize=10, color="#FF5722")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelcolor="#FF5722")

        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_title("PointNet++ MSG Training Curve\n"
                      "Semantic Segmentation (tree / no-tree) — FOR-instance val set",
                      fontsize=11, fontweight="bold")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
        ax1.spines[["top"]].set_visible(False)

        fig.tight_layout()
        out = FIGURES_DIR / "04_metrics" / "pointnet2_training_curve.png"
        return save_fig(fig, out)
    except Exception as e:
        fig_failed(name, e)
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# HTML REPORT
# ──────────────────────────────────────────────────────────────────────────────

def _img_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def generate_html():
    """Generate self-contained HTML with all figures embedded as base64."""
    name = "html_report"
    try:
        rf_df   = pd.read_csv(OUTPUT_DIR / "results" / "rf_metrics_val.csv")
        base_df = pd.read_csv(OUTPUT_DIR / "results" / "baseline_metrics_val.csv")
        pointnet2_df  = pd.read_csv(OUTPUT_DIR / "results" / "pointnet2_metrics_val.csv")

        def img_tag(path: Path, caption: str, width: str = "100%") -> str:
            if not path.exists():
                return f'<p style="color:red">[Figure missing: {path.name}]</p>'
            b64 = _img_b64(path)
            return (f'<figure style="margin:16px 0">'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="width:{width};border:1px solid #ddd;" />'
                    f'<figcaption style="font-size:12px;color:#555;margin-top:4px">'
                    f'{caption}</figcaption></figure>')

        def row_style(i): return "background:#f9f9f9" if i % 2 == 0 else ""

        # Metric table helper
        def metric_table(df, method_label, sem=True):
            cols_inst = ["institution","f1","precision","recall","over_seg","under_seg","n_gt_trees","n_pred_trees"]
            if sem and "sem_miou" in df.columns:
                cols_inst.append("sem_miou")
            available = [c for c in cols_inst if c in df.columns]
            sub = df[available].copy()
            html = '<table style="border-collapse:collapse;width:100%;font-size:12px">'
            html += '<tr style="background:#E3F2FD">'
            for c in available:
                html += f'<th style="padding:6px;border:1px solid #ddd">{c}</th>'
            html += '</tr>'
            for i, (_, r) in enumerate(sub.iterrows()):
                html += f'<tr style="{row_style(i)}">'
                for c in available:
                    v = r[c]
                    fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
                    html += f'<td style="padding:5px;border:1px solid #ddd;text-align:right">{fmt}</td>'
                html += '</tr>'
            html += '</table>'
            return html

        # Summary table data
        summary_rows = [
            ("Baseline (A)", "0.211", "0.268", "0.192", "0.721", "0.280", "—"),
            ("RF+WS (B)",    "0.344", "0.461", "0.326", "0.674", "0.181", "0.994"),
            ("PointNet2+WS (C)",   "0.288", "0.388", "0.263", "0.737", "0.162", "0.988"),
        ]
        by_inst_rows = [
            ("CULS",  "0.471", "0.732", "0.635", "PointNet2+WS ✓"),
            ("NIBIO", "0.023", "0.129", "0.085", "RF+WS ✓"),
            ("SCION", "0.065", "0.000", "0.000", "Baseline"),
        ]

        FD = FIGURES_DIR

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ITS Comparison — Paper Figures</title>
<style>
body {{font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px;
      background:#fff;color:#222;}}
h1 {{color:#1565C0;border-bottom:2px solid #1565C0;padding-bottom:8px}}
h2 {{color:#1976D2;margin-top:32px;border-left:4px solid #1976D2;padding-left:10px}}
h3 {{color:#333;margin-top:20px}}
table {{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}}
th {{background:#E3F2FD;padding:8px;border:1px solid #ddd;text-align:center}}
td {{padding:6px 8px;border:1px solid #ddd}}
.badge {{display:inline-block;padding:2px 8px;border-radius:10px;
         font-size:11px;font-weight:bold;color:#fff}}
.toc {{background:#f5f5f5;padding:16px;border-radius:6px;margin:16px 0}}
.toc a {{color:#1565C0;text-decoration:none;display:block;padding:2px 0}}
.toc a:hover {{text-decoration:underline}}
.note {{background:#FFF8E1;border-left:4px solid #FFC107;padding:10px 14px;
        margin:12px 0;font-size:13px}}
figcaption {{font-size:12px;color:#555;margin-top:4px;font-style:italic}}
</style>
</head>
<body>
<h1>ITS Pipeline Comparison — Paper Figures</h1>
<p style="color:#555">
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
  Val set (5 plots): CULS×2, NIBIO×2, SCION×1 &nbsp;|&nbsp;
  Dataset: FOR-instance (Puliti et al. 2023)
</p>

<div class="toc">
<strong>Table of Contents</strong>
<a href="#sec0">0. Executive Summary</a>
<a href="#sec1">1. Dataset Overview</a>
<a href="#sec2">2. Semantic Results</a>
<a href="#sec3">3. Instance Results</a>
<a href="#sec4">4. Metric Comparisons</a>
<a href="#sec5">5. Full Metric Tables</a>
<a href="#sec6">6. Methodological Notes</a>
</div>

<!-- ─────────────────────────── SECTION 0 ─────────────────────────── -->
<h2 id="sec0">0. Executive Summary</h2>

<h3>Overall Results (Val Set, mean over 5 plots)</h3>
<table>
<tr><th>Method</th><th>F1</th><th>Precision</th><th>Recall</th>
    <th>Under-seg</th><th>Over-seg</th><th>sem_mIoU</th></tr>
{''.join(f'<tr><td><b>{r[0]}</b></td>'+
         ''.join(f'<td style="text-align:center">{v}</td>' for v in r[1:])+'</tr>'
         for r in summary_rows)}
</table>

<h3>F1 by Institution</h3>
<table>
<tr><th>Institution</th><th>Baseline</th><th>RF+WS</th><th>PointNet2+WS</th><th>Best</th></tr>
{''.join(f'<tr><td><b>{r[0]}</b></td>'+
         ''.join(f'<td style="text-align:center">{v}</td>' for v in r[1:])+'</tr>'
         for r in by_inst_rows)}
</table>

<div class="note">
<b>Central finding:</b> Semantic preprocessing quality (RF mIoU=0.994, PointNet2 mIoU=0.988)
does not translate into instance segmentation quality in dense forests (NIBIO/SCION:
under-seg &gt;0.95 for all methods). The Watershed 3D algorithm is the bottleneck.
Semantic preprocessing only helps in open forests (CULS).
</div>

<!-- ─────────────────────────── SECTION 1 ─────────────────────────── -->
<h2 id="sec1">1. Dataset Overview</h2>
{img_tag(FD/"01_dataset_overview"/"semantic_gt_overview.png",
         "Figure 1.1 — Ground-truth semantic labels for three representative plots. "
         "Top row: top-down view. Bottom row: lateral cross-section (Y = center ±5m).")}
{img_tag(FD/"01_dataset_overview"/"dataset_stats.png",
         "Figure 1.2 — Dataset statistics: point density, GT tree counts, "
         "semantic IoU by class, and density vs. instance F1 correlation.")}

<!-- ─────────────────────────── SECTION 2 ─────────────────────────── -->
<h2 id="sec2">2. Semantic Results</h2>
{img_tag(FD/"02_semantic_results"/"semantic_comparison_plot_1.png",
         "Figure 2.1a — CULS plot_1: GT semantic labels vs. RF and PointNet2 predictions (top view).")}
{img_tag(FD/"02_semantic_results"/"semantic_comparison_plot_2.png",
         "Figure 2.1b — NIBIO plot_2: GT semantic labels vs. RF and PointNet2 predictions (top view).")}
{img_tag(FD/"02_semantic_results"/"semantic_error_map_plot_1.png",
         "Figure 2.2a — CULS plot_1: Semantic error map (TP/TN/FP/FN) for RF and PointNet2.")}
{img_tag(FD/"02_semantic_results"/"semantic_error_map_plot_2.png",
         "Figure 2.2b — NIBIO plot_2: Semantic error map for RF and PointNet2.")}
{img_tag(FD/"02_semantic_results"/"confusion_matrices.png",
         "Figure 2.3 — Recall-normalised confusion matrices for RF and PointNet2 on CULS and NIBIO.")}

<!-- ─────────────────────────── SECTION 3 ─────────────────────────── -->
<h2 id="sec3">3. Instance Results</h2>
{img_tag(FD/"03_instance_results"/"instance_comparison_culs_plot1.png",
         "Figure 3.1 — CULS plot_1 (open forest, 6 GT trees): instance segmentation "
         "comparison across all three methods.")}
{img_tag(FD/"03_instance_results"/"instance_comparison_nibio_plot2.png",
         "Figure 3.2 — NIBIO plot_2 (dense forest, 40 GT trees): Watershed under-segmentation "
         "is the dominant failure mode for all methods.")}
{img_tag(FD/"03_instance_results"/"instance_3d_lateral_culs_plot1.png",
         "Figure 3.3 — CULS plot_1: lateral (XZ) view of instance segmentation. "
         "Left=GT, Centre=RF, Right=PointNet2.")}
{img_tag(FD/"03_instance_results"/"underseg_illustration.png",
         "Figure 3.4 — NIBIO plot_2: 20×20m zoom illustrating severe under-segmentation. "
         "Dashed circles highlight GT trees merged into a single predicted segment.")}

<!-- ─────────────────────────── SECTION 4 ─────────────────────────── -->
<h2 id="sec4">4. Metric Comparisons</h2>
{img_tag(FD/"04_metrics"/"f1_by_institution.png",
         "Figure 4.1 — Instance F1 by institution and method. "
         "Error bars show std across plots within each institution.")}
{img_tag(FD/"04_metrics"/"semantic_vs_instance_decoupling.png",
         "Figure 4.2 — Semantic mIoU vs. instance F1. "
         "The shaded region highlights the decoupling zone: high semantic quality "
         "coexists with near-zero instance F1 in dense forests.")}
{img_tag(FD/"04_metrics"/"underseg_overseg_comparison.png",
         "Figure 4.3 — Under- vs. over-segmentation by plot and method.")}
{img_tag(FD/"04_metrics"/"pointnet2_training_curve.png",
         "Figure 4.4 — PointNet++ MSG training curve. "
         "Best val mIoU = 0.9914 at epoch 87.")}

<!-- ─────────────────────────── SECTION 5 ─────────────────────────── -->
<h2 id="sec5">5. Full Metric Tables</h2>

<h3>Baseline (Method A)</h3>
{metric_table(base_df, "Baseline", sem=False)}

<h3>RF + Watershed (Method B)</h3>
{metric_table(rf_df, "RF+WS", sem=True)}

<h3>PointNet2 + Watershed (Method C)</h3>
{metric_table(pointnet2_df, "PointNet2+WS", sem=True)}

<!-- ─────────────────────────── SECTION 6 ─────────────────────────── -->
<h2 id="sec6">6. Methodological Notes</h2>
<table>
<tr><th>Component</th><th>Decision</th><th>Justification</th></tr>
<tr><td>Watershed sigma</td><td>gaussian_sigma=0.5</td><td>Moderate smoothing of 3D density grid; preserves crown structure</td></tr>
<tr><td>Min crown radius</td><td>min_crown_radius_m=1.0</td><td>Empirical on val set</td></tr>
<tr><td>RF features</td><td>28 = 14×2 scales (k=20, k=50)</td><td>Weinmann et al. (2017) ISPRS</td></tr>
<tr><td>RF class weight</td><td>balanced</td><td>Severe tree/non-tree imbalance</td></tr>
<tr><td>PointNet2 architecture</td><td>MSG over SSG</td><td>Qi et al. (2017) — variable density 10×</td></tr>
<tr><td>PointNet2 input features</td><td>5 channels (XYZ + HAG + intensity)</td><td>Wang et al. (2023) Remote Sens.</td></tr>
<tr><td>PointNet2 training</td><td>AMP bf16 (no GradScaler)</td><td>Apple Silicon MPS / unified memory</td></tr>
<tr><td>PointNet2 inference</td><td>Global random subsampling, n_passes=N/8192×3</td><td>Matches training distribution; sliding window causes 14× scale mismatch</td></tr>
<tr><td>KNN coverage</td><td>scipy.cKDTree k=3 (1/dist weighted)</td><td>Closes 14% coverage gap; sem_mIoU 0.908→0.988</td></tr>
<tr><td>Instance metric</td><td>IoU threshold=0.5</td><td>Standard ForAINet and SegmentAnyTree</td></tr>
</table>

<div class="note">
<b>NaN training bug (fixed):</b> In AMP fp16, <code>1e-8</code> underflows to 0 in
<code>dist_recip = 1/(dists + 1e-8)</code> when query points exactly match centroids
(guaranteed in FP1 since the 8192-point sample contains the 1024 SA1 centroids).
Fix: <code>dists.float()</code> before epsilon. Confirmed in pointnet2_utils.py:300.
</div>

</body></html>"""

        out_path = FIGURES_DIR / "paper_summary.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        GENERATED.append(out_path)
        log.info(f"  HTML saved: {out_path}")
    except Exception as e:
        fig_failed(name, e)
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    steps = [
        # Group 1
        ("1.1 semantic GT overview",     lambda: fig_1_1_semantic_gt_overview(cfg)),
        ("1.2 dataset stats",            lambda: fig_1_2_dataset_stats(cfg)),
        # Group 4 (fast — CSV only)
        ("4.1 F1 by institution",        fig_4_1_f1_by_institution),
        ("4.2 decoupling scatter",       fig_4_2_decoupling_scatter),
        ("4.3 under/over segmentation",  fig_4_3_underseg_overseg),
        ("4.4 PointNet2 training curve",       fig_4_4_training_curve),
        # Group 2
        ("2.1 semantic comparison",      lambda: fig_2_1_semantic_comparison(cfg)),
        ("2.2 semantic error maps",      lambda: fig_2_2_semantic_error_maps(cfg)),
        ("2.3 confusion matrices",       lambda: fig_2_3_confusion_matrices(cfg)),
        # Group 3
        ("3.1 instance CULS plot1",      lambda: fig_3_1_instance_culs_plot1(cfg)),
        ("3.2 instance NIBIO plot2",     lambda: fig_3_2_instance_nibio_plot2(cfg)),
        ("3.3 3D lateral CULS plot1",    lambda: fig_3_3_instance_3d_lateral(cfg)),
        ("3.4 underseg illustration",    lambda: fig_3_4_underseg_illustration(cfg)),
        # HTML
        ("HTML report",                  generate_html),
    ]

    total = len(steps)
    for i, (label, fn) in enumerate(steps, 1):
        log.info(f"\n[{i}/{total}] {label}")
        fn()

    log.info("\n" + "="*60)
    log.info(f"Figures generated: {len(GENERATED)}/{total}")
    if FAILED:
        log.info(f"Figures FAILED ({len(FAILED)}):")
        for f in FAILED:
            log.info(f"  - {f}")
    else:
        log.info("All figures generated successfully.")

    total_size = sum(p.stat().st_size for p in GENERATED if p.exists()) / 1e6
    log.info(f"Total size: {total_size:.1f} MB")
    log.info(f"HTML: {FIGURES_DIR / 'paper_summary.html'}")
    log.info("="*60)


if __name__ == "__main__":
    main()
