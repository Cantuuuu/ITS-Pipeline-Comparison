"""
Genera el diagrama del pipeline ITS (diseño factorial 3x2) como PNG.
Ejecutar desde la raíz del proyecto:
    python output/figures/generate_pipeline.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.5)
ax.axis("off")

# ---------- Colores ----------
C_INPUT  = "#2C3E50"
C_STAGE1 = "#2980B9"
C_STAGE2 = "#27AE60"
C_FLOW   = "#ECF0F1"
C_BEST   = "#E74C3C"
C_TEXT   = "white"
C_DARK   = "#2C3E50"
ALPHA    = 0.92

def box(ax, x, y, w, h, label, sublabel="", color=C_STAGE1,
        fontsize=9, subsize=7.5, radius=0.25):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=color, edgecolor="white", linewidth=1.5, alpha=ALPHA,
        zorder=3
    )
    ax.add_patch(patch)
    cy = y + h / 2 + (0.15 if sublabel else 0)
    ax.text(x + w/2, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=C_TEXT, zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel, ha="center", va="center",
                fontsize=subsize, color=C_TEXT, alpha=0.85, zorder=4,
                style="italic")

def arrow(ax, x1, y1, x2, y2, color="#7F8C8D", lw=1.5, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)

# ── Entrada ──────────────────────────────────────────────────────────────────
box(ax, 0.15, 2.15, 1.5, 1.2, "ULS Point", "Cloud", color=C_INPUT,
    fontsize=9, subsize=8)

# ── ETAPA 1: Preprocesado semántico ──────────────────────────────────────────
ax.text(3.3, 5.1, "Etapa 1 — Preprocesado semántico",
        ha="center", va="center", fontsize=9, fontweight="bold", color=C_DARK)

stage1 = [
    (2.3, 3.65, "Sin preproc.", "(baseline)", "#7F8C8D"),
    (2.3, 2.25, "Random\nForest", "27 geom. features", C_STAGE1),
    (2.3, 0.85, "PointNet++\nMSG", "5D input, 8192 pts", "#8E44AD"),
]
for (x, y, lbl, sub, col) in stage1:
    box(ax, x, y, 2.0, 1.1, lbl, sub, color=col, fontsize=9)
    arrow(ax, 1.65, 2.75, x, y + 0.55)

# ── ETAPA 2: Segmentador de instancias ───────────────────────────────────────
ax.text(7.2, 5.1, "Etapa 2 — Segmentador de instancias",
        ha="center", va="center", fontsize=9, fontweight="bold", color=C_DARK)

box(ax, 5.9, 3.2, 2.55, 1.1, "Watershed 3D", "Seeding por CHM",
    color=C_STAGE2, fontsize=9)
box(ax, 5.9, 1.65, 2.55, 1.1, "Watershed 3D", "Seeding por Densidad",
    color="#16A085", fontsize=9)

# Flechas stage1 → stage2 (CHM y Density)
# No preproc → CHM
arrow(ax, 4.3, 4.2,  5.9, 3.9)
# RF → CHM
arrow(ax, 4.3, 2.8,  5.9, 3.5)
# PN++ → CHM
arrow(ax, 4.3, 1.4,  5.9, 3.3)
# RF → Density
arrow(ax, 4.3, 2.6,  5.9, 2.1)
# PN++ → Density
arrow(ax, 4.3, 1.2,  5.9, 1.9)

# ── FLUJOS ───────────────────────────────────────────────────────────────────
ax.text(10.15, 5.1, "Flujos evaluados",
        ha="center", va="center", fontsize=9, fontweight="bold", color=C_DARK)

flows = [
    (9.15, 4.35, "Flujo A", "Sin preproc. + CHM",    "#7F8C8D"),
    (9.15, 3.35, "Flujo B", "RF + CHM",               C_STAGE1),
    (9.15, 2.35, "Flujo C", "PN++ + CHM",             "#8E44AD"),
    (9.15, 1.35, "Flujo D", "RF + Density",           C_BEST),
    (9.15, 0.35, "Flujo E", "PN++ + Density",         "#C0392B"),
]
for (x, y, lbl, sub, col) in flows:
    box(ax, x, y, 2.2, 0.85, lbl, sub, color=col, fontsize=8.5, subsize=7)

# Flechas stage2 → flows (CHM → A,B,C; Density → D,E)
for fy in [4.35+0.425, 3.35+0.425, 2.35+0.425]:
    arrow(ax, 8.45, 3.75, 9.15, fy, color="#27AE60")
for fy in [1.35+0.425, 0.35+0.425]:
    arrow(ax, 8.45, 2.2, 9.15, fy, color="#16A085")

# Estrella en Flujo D (mejor resultado)
ax.text(11.55, 1.775, "★ Best\nF1=0.209", ha="center", va="center",
        fontsize=7.5, color=C_BEST, fontweight="bold", zorder=5)

# ── Separadores verticales ────────────────────────────────────────────────────
for xv in [2.15, 5.75, 9.0]:
    ax.axvline(xv, color="#BDC3C7", lw=0.8, linestyle="--", alpha=0.5, zorder=1)

plt.tight_layout(pad=0.3)
out = "output/figures/pipeline_diagram.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Guardado: {out}")
