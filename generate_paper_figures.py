"""
Genera todas las figuras necesarias para el paper desde los CSVs de resultados.
Las imágenes se guardan en assets/paper/
Ejecutar: python generate_paper_figures.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
colors = {
    'baseline': '#7F8C8D',
    'rf': '#2980B9',
    'pointnet2': '#8E44AD',
    'rf_density': '#E74C3C',
    'pointnet2_density': '#C0392B'
}

methods_order = ['baseline', 'rf', 'pointnet2', 'rf_density', 'pointnet2_density']
method_labels = {
    'baseline': 'A: Sin preproc.',
    'rf': 'B: RF + CHM',
    'pointnet2': 'C: PN++ + CHM',
    'rf_density': 'D: RF + Density',
    'pointnet2_density': 'E: PN++ + Density'
}

output_dir = Path('assets/paper')
output_dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. RESULTADOS AGREGADOS (TEST) - F1 y Métricas por método
# ─────────────────────────────────────────────────────────────────────────────

df_test = pd.read_csv('results/comparison_table_test.csv')
agg = df_test.groupby('method')[['f1', 'precision', 'recall', 'coverage',
                                   'sem_miou', 'sem_seg_f1']].mean()
agg = agg.reindex([m for m in methods_order if m in agg.index])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# F1 de instancia
x = np.arange(len(agg))
bars = ax1.bar(x, agg['f1'], color=[colors.get(m, '#95A5A6') for m in agg.index],
               alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('F1 (Instance Segmentation)', fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([method_labels.get(m, m) for m in agg.index], rotation=15, ha='right')
ax1.set_ylim(0, 0.25)
ax1.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.3f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Comparación de métricas semánticas
x = np.arange(len(agg))
w = 0.25
ax2.bar(x - w, agg['sem_miou'], w, label='Sem. mIoU', color='#3498DB', alpha=0.8)
ax2.bar(x, agg['sem_seg_f1'], w, label='Sem. F1', color='#2ECC71', alpha=0.8)
ax2.bar(x + w, agg['f1'], w, label='Inst. F1', color='#E74C3C', alpha=0.8)
ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([method_labels.get(m, m) for m in agg.index], rotation=15, ha='right')
ax2.legend(loc='lower right')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(output_dir / 'fig_01_aggregated_results.png', dpi=150, bbox_inches='tight')
print("[OK] fig_01_aggregated_results.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 2. F1 POR SITIO (INSTITUCIÓN) - Comparativa entre métodos
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
pivot = df_test.pivot_table(values='f1', index='institution', columns='method', aggfunc='mean')
pivot = pivot[[m for m in methods_order if m in pivot.columns]]
pivot.columns = [method_labels.get(m, m) for m in pivot.columns]

pivot.plot(kind='bar', ax=ax, color=[colors.get(m, '#95A5A6') for m in methods_order
                                      if m in df_test['method'].unique()],
           width=0.8, edgecolor='black', linewidth=1)
ax.set_ylabel('F1 (Instance Segmentation)', fontsize=11, fontweight='bold')
ax.set_xlabel('Forest Type (Institution)', fontsize=11, fontweight='bold')
ax.set_title('Instance Segmentation F1 Score by Forest Type', fontsize=12, fontweight='bold')
ax.legend(title='Method', loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(output_dir / 'fig_02_f1_by_institution.png', dpi=150, bbox_inches='tight')
print("[OK] fig_02_f1_by_institution.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3. DESACOPLAMIENTO SEMÁNTICO-INSTANCIA
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
methods_unique = [m for m in methods_order if m in df_test['method'].unique()]
scatter_data = df_test.groupby('method')[['sem_miou', 'f1']].mean().reindex(methods_unique)

for method in methods_unique:
    x, y = scatter_data.loc[method, 'sem_miou'], scatter_data.loc[method, 'f1']
    ax.scatter(x, y, s=300, alpha=0.7, color=colors.get(method, '#95A5A6'),
              edgecolor='black', linewidth=2, label=method_labels.get(method, method))
    ax.annotate(method_labels.get(method, method).split(':')[0],
               xy=(x, y), xytext=(5, 5), textcoords='offset points',
               fontsize=9, fontweight='bold')

ax.set_xlabel('Semantic Quality (mIoU)', fontsize=11, fontweight='bold')
ax.set_ylabel('Instance Segmentation (F1)', fontsize=11, fontweight='bold')
ax.set_title('Decoupling: Semantic vs Instance Quality', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.97, 1.0)
ax.set_ylim(0, 0.25)
plt.tight_layout()
plt.savefig(output_dir / 'fig_03_semantic_instance_decoupling.png', dpi=150, bbox_inches='tight')
print("[OK] fig_03_semantic_instance_decoupling.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4. PRECISIÓN vs RECALL (Factory plot)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
methods_unique = [m for m in methods_order if m in df_test['method'].unique()]
pr_data = df_test.groupby('method')[['precision', 'recall']].mean().reindex(methods_unique)

for method in methods_unique:
    p, r = pr_data.loc[method, 'precision'], pr_data.loc[method, 'recall']
    ax.scatter(r, p, s=300, alpha=0.7, color=colors.get(method, '#95A5A6'),
              edgecolor='black', linewidth=2, label=method_labels.get(method, method))
    ax.annotate(method_labels.get(method, method).split(':')[0],
               xy=(r, p), xytext=(5, 5), textcoords='offset points',
               fontsize=9, fontweight='bold')

ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
ax.set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 0.4)
ax.set_ylim(0, 0.5)
plt.tight_layout()
plt.savefig(output_dir / 'fig_04_precision_recall.png', dpi=150, bbox_inches='tight')
print("[OK] fig_04_precision_recall.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 5. FACTORIAL ANALYSIS: Efecto de preprocesado vs seeding
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

# Agrupar por preprocesado y tipo de seeding
df_test['preproc'] = df_test['method'].map({
    'baseline': 'Sin preproc',
    'rf': 'RF',
    'pointnet2': 'PointNet++',
    'rf_density': 'RF',
    'pointnet2_density': 'PointNet++'
})
df_test['seeding'] = df_test['method'].map({
    'baseline': 'CHM',
    'rf': 'CHM',
    'pointnet2': 'CHM',
    'rf_density': 'Density',
    'pointnet2_density': 'Density'
})

pivot = df_test.pivot_table(values='f1', index='preproc', columns='seeding', aggfunc='mean')
pivot = pivot[['CHM', 'Density']]

x = np.arange(len(pivot))
w = 0.35
bars1 = ax.bar(x - w/2, pivot['CHM'], w, label='CHM Seeding', alpha=0.8,
              color='#3498DB', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + w/2, pivot['Density'], w, label='Density Seeding', alpha=0.8,
              color='#E74C3C', edgecolor='black', linewidth=1.5)

ax.set_ylabel('F1 (Instance Segmentation)', fontsize=11, fontweight='bold')
ax.set_xlabel('Semantic Preprocessing', fontsize=11, fontweight='bold')
ax.set_title('Factorial Analysis: Impact of Preprocessing vs Seeding', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pivot.index)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.3f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig_05_factorial_analysis.png', dpi=150, bbox_inches='tight')
print("[OK] fig_05_factorial_analysis.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 6. OVER-SEGMENTATION vs UNDER-SEGMENTATION por método
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))
methods_unique = [m for m in methods_order if m in df_test['method'].unique()]
seg_data = df_test.groupby('method')[['over_seg', 'under_seg']].mean().reindex(methods_unique)

x = np.arange(len(seg_data))
w = 0.35
bars1 = ax.bar(x - w/2, seg_data['over_seg'], w, label='Over-segmentation',
              alpha=0.8, color='#F39C12', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + w/2, seg_data['under_seg'], w, label='Under-segmentation',
              alpha=0.8, color='#E67E22', edgecolor='black', linewidth=1.5)

ax.set_ylabel('Segmentation Error Rate', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([method_labels.get(m, m) for m in seg_data.index], rotation=15, ha='right')
ax.set_title('Crown Fragmentation: Over vs Under-segmentation', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'fig_06_segmentation_errors.png', dpi=150, bbox_inches='tight')
print("[OK] fig_06_segmentation_errors.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 7. MATRIZ DE CONFUSIÓN - Comparar mejor vs peor método
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

methods_to_show = ['baseline', 'rf', 'rf_density', 'pointnet2']
for idx, method in enumerate(methods_to_show):
    ax = axes[idx // 2, idx % 2]

    method_data = df_test[df_test['method'] == method]
    tp = method_data['TP'].sum()
    fp = method_data['FP'].sum()
    fn = method_data['FN'].sum()
    tn = 1000  # Valor ficticio para visualización

    confusion = np.array([[tp, fp], [fn, tn]])
    im = ax.imshow(confusion / confusion.sum(), cmap='Blues', aspect='auto')

    # Añadir texto
    ax.text(0, 0, f'{tp}\n(TP)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(1, 0, f'{fp}\n(FP)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(0, 1, f'{fn}\n(FN)', ha='center', va='center', fontsize=11, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted +', 'Predicted -'])
    ax.set_yticklabels(['Actual +', 'Actual -'])
    ax.set_title(method_labels.get(method, method), fontsize=11, fontweight='bold')

plt.suptitle('Confusion Matrices: Detection Errors by Method', fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / 'fig_07_confusion_matrices.png', dpi=150, bbox_inches='tight')
print("[OK] fig_07_confusion_matrices.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 8. SEMANTIC QUALITY HEATMAP - por sitio y método
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))
pivot_sem = df_test.pivot_table(values='sem_miou', index='institution', columns='method', aggfunc='mean')
pivot_sem = pivot_sem[[m for m in methods_order if m in pivot_sem.columns]]

sns.heatmap(pivot_sem, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.96, vmax=1.0,
           ax=ax, cbar_kws={'label': 'Semantic mIoU'}, linewidths=0.5, linecolor='gray')
ax.set_title('Semantic Segmentation Quality by Forest Type', fontsize=12, fontweight='bold')
ax.set_xlabel('Method', fontsize=11, fontweight='bold')
ax.set_ylabel('Forest Type', fontsize=11, fontweight='bold')
ax.set_xticklabels([method_labels.get(m, m).split(':')[0] for m in pivot_sem.columns], rotation=15, ha='right')

plt.tight_layout()
plt.savefig(output_dir / 'fig_08_semantic_heatmap.png', dpi=150, bbox_inches='tight')
print("[OK] fig_08_semantic_heatmap.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 9. INSTANCE RESULTS HEATMAP - por sitio y método
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))
pivot_inst = df_test.pivot_table(values='f1', index='institution', columns='method', aggfunc='mean')
pivot_inst = pivot_inst[[m for m in methods_order if m in pivot_inst.columns]]

sns.heatmap(pivot_inst, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=0.3,
           ax=ax, cbar_kws={'label': 'Instance F1'}, linewidths=0.5, linecolor='gray')
ax.set_title('Instance Segmentation Results by Forest Type', fontsize=12, fontweight='bold')
ax.set_xlabel('Method', fontsize=11, fontweight='bold')
ax.set_ylabel('Forest Type', fontsize=11, fontweight='bold')
ax.set_xticklabels([method_labels.get(m, m).split(':')[0] for m in pivot_inst.columns], rotation=15, ha='right')

plt.tight_layout()
plt.savefig(output_dir / 'fig_09_instance_heatmap.png', dpi=150, bbox_inches='tight')
print("[OK] fig_09_instance_heatmap.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 10. COVERAGE METRIC - Detección de árboles (con/sin sobresegmentación)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))
methods_unique = [m for m in methods_order if m in df_test['method'].unique()]
cov_data = df_test.groupby('method')[['coverage', 'recall']].mean().reindex(methods_unique)

x = np.arange(len(cov_data))
w = 0.35
bars1 = ax.bar(x - w/2, cov_data['coverage'], w, label='Coverage (≥1 prediction)',
              alpha=0.8, color='#27AE60', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + w/2, cov_data['recall'], w, label='Recall (1-to-1 match, IoU≥0.5)',
              alpha=0.8, color='#2980B9', edgecolor='black', linewidth=1.5)

ax.set_ylabel('Detection Rate', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([method_labels.get(m, m) for m in cov_data.index], rotation=15, ha='right')
ax.set_title('Tree Detection: Coverage vs Recall (Over-segmentation Indicator)',
            fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.4)

# Add value labels and gap indicator
for i, method in enumerate(cov_data.index):
    cov = cov_data.loc[method, 'coverage']
    rec = cov_data.loc[method, 'recall']
    gap = cov - rec
    if gap > 0.05:
        ax.annotate(f'{gap:.2f}', xy=(i, (cov + rec)/2), ha='center', va='center',
                   fontsize=8, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig(output_dir / 'fig_10_coverage_analysis.png', dpi=150, bbox_inches='tight')
print("[OK] fig_10_coverage_analysis.png")
plt.close()

print("\n[SUCCESS] All figures generated successfully in assets/paper/")
print(f"   Total: 10 figures")
