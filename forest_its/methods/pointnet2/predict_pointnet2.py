"""
Inferencia PointNet++ MSG por submuestreo global del plot.

Estrategia de inferencia:
  La red fue entrenada con submuestreo aleatorio GLOBAL del plot completo
  (ForInstanceDataset: 8192 pts de todo el plot, normalización por radio
  del sample). La inferencia DEBE replicar esta misma estrategia.

  Usar sliding window de bloques pequeños (10m) introduce un mismatch de
  escala ~14× en el espacio normalizado (las SA layers usan radios
  normalizados relativos a la escala del input), causando predicciones
  erróneas.

  Estrategia correcta (matching training):
    1. Submuestrear aleatoriamente num_points=8192 del plot completo
    2. Normalizar exactamente igual que en training:
         - Pre-centrado XY en centroide de puntos válidos (igual __init__)
         - Per-sample: centrar en centroide del sample + normalizar por radio
    3. Repetir n_passes veces para aumentar cobertura de puntos
    4. Acumular probabilidades por punto; promediar al final
    5. Puntos no cubiertos → clase mayoritaria del plot

  Cobertura teórica con n_passes = clip(3·N / N_s, 50, 1000):
    P(un punto no sea muestreado en una pasada) ≈ exp(-N_s/N)
    P(un punto no sea muestreado tras n_passes) ≈ exp(-3) ≈ 5%
    → cobertura por punto ≈ 95% antes del KNN final.
    La interpolación KNN sobre puntos no cubiertos lleva la cobertura
    efectiva a >99% en todos los plots observados.

Referencia:
  Qi et al. (2017) PointNet++: el modelo fue diseñado para operar sobre
  nubes globales (sub)muestreadas, no sobre bloques locales pequeños.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from forest_its.data.dataset import load_las, get_binary_labels
from forest_its.preprocessing.normalize_height import process_plot
from forest_its.methods.pointnet2.model_msg import PointNet2SemSegMSG


# Número de samples por forward pass en inferencia (>1 → eficiencia GPU).
# 16 aprovecha la memoria unificada del M4 Pro; reducir si hay OOM en CUDA 6 GB.
_INFER_BATCH = 16


def compute_coverage(prob_count: np.ndarray) -> dict:
    """
    Diagnóstico de cobertura tras el loop de subsampling.

    Args:
        prob_count: (N,) int32 — cuántas veces fue cubierto cada punto.

    Returns:
        Dict con total_points, covered_points, coverage_pct, uncovered_count.
    """
    total = len(prob_count)
    covered = int((prob_count > 0).sum())
    return {
        "total_points": total,
        "covered_points": covered,
        "uncovered_count": total - covered,
        "coverage_pct": 100.0 * covered / total if total > 0 else 0.0,
    }


def assign_uncovered_points(
    all_xyz: np.ndarray,
    pred_proba: np.ndarray,
    covered_mask: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """
    Asigna probabilidades a puntos no cubiertos por KNN ponderado (1/distancia)
    desde los puntos cubiertos más cercanos.

    Args:
        all_xyz:      (N, 3) float32 — coordenadas XYZ pre-centradas.
        pred_proba:   (N, 2) float32 — probabilidades acumuladas (cubiertos OK,
                      no cubiertos tienen valores arbitrarios que serán sobreescritos).
        covered_mask: (N,) bool — True si el punto fue cubierto ≥1 vez.
        k:            Número de vecinos cubiertos más cercanos para interpolar.

    Returns:
        pred_proba actualizado in-place: (N, 2) float32.
    """
    uncovered_mask = ~covered_mask
    n_uncovered = uncovered_mask.sum()
    if n_uncovered == 0:
        return pred_proba

    covered_xyz = all_xyz[covered_mask]
    covered_proba = pred_proba[covered_mask]   # (n_covered, 2)
    uncovered_xyz = all_xyz[uncovered_mask]

    tree = cKDTree(covered_xyz)
    k_actual = min(k, len(covered_xyz))
    dists, idxs = tree.query(uncovered_xyz, k=k_actual)   # (n_uncovered, k)

    if k_actual == 1:
        # query devuelve scalars cuando k=1
        dists = dists[:, None]
        idxs = idxs[:, None]

    # Pesos 1/distancia; distancia exactamente 0 → peso 1, resto 0
    weights = np.where(dists == 0, 1.0, 1.0 / (dists + 1e-8)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)           # (n_uncovered, k)

    # Interpolación ponderada de probabilidades
    interp = np.einsum("nk,nkc->nc", weights, covered_proba[idxs])  # (n_uncovered, 2)
    pred_proba[uncovered_mask] = interp

    return pred_proba


def predict_plot_pointnet2(las_data: dict, model, cfg, device) -> tuple:
    """
    Inferencia PointNet++ sobre un plot completo.

    Replica exactamente la estrategia de ForInstanceDataset:
      - Pre-centrado XY sobre puntos válidos (igual que __init__)
      - Submuestreo aleatorio global del plot entero (igual que __getitem__)
      - Normalización per-sample: centroide + radio del sample
      - HAG normalizado a [0, 1]; intensidad ya en [0, 1]

    Args:
        las_data: Dict de load_las() con "hag" ya computado vía process_plot().
        model:    PointNet2SemSegMSG en eval mode.
        cfg:      OmegaConf config.
        device:   torch.device.

    Returns:
        pred_binary: (N,) int32 — clase predicha (0=no-árbol, 1=árbol).
        pred_proba:  (N, 2) float32 — probabilidades promediadas.
    """
    from torch.amp import autocast

    xyz_all = las_data["xyz"].astype(np.float32)        # (N, 3)
    hag_all = las_data["hag"].astype(np.float32)        # (N,)
    intensity_all = las_data["intensity"].astype(np.float32)  # (N,)
    N = len(xyz_all)

    num_points = int(cfg.pointnet2.num_points)          # 8192
    hag_max = float(cfg.preprocessing.hag_max)          # 50.0

    # RNG local determinista para el submuestreo de inferencia. Se usa
    # np.random.default_rng(seed) en vez de np.random.choice(...) global
    # para garantizar que corridas repetidas de inferencia sobre el mismo
    # plot produzcan exactamente las mismas predicciones (ver §3.3 del paper).
    rng = np.random.default_rng(int(cfg.data.random_state))

    # Pre-centrado XY sobre puntos válidos (replica ForInstanceDataset.__init__)
    binary_labels = get_binary_labels(las_data["classification"])
    valid_mask = binary_labels != -1
    if valid_mask.any():
        xy_center = xyz_all[valid_mask, :2].mean(axis=0)
    else:
        xy_center = xyz_all[:, :2].mean(axis=0)

    xyz_c = xyz_all.copy()
    xyz_c[:, 0] -= xy_center[0]
    xyz_c[:, 1] -= xy_center[1]

    # Normalizar HAG a [0, 1] (replica ForInstanceDataset.__getitem__)
    hag_norm_all = np.clip(hag_all / hag_max, 0.0, 1.0)

    # Número de passes: cada punto esperado cubierto ~3 veces en media
    # Cap en 1000 para evitar tiempos excesivos en plots muy grandes
    n_passes = min(1000, max(50, int(N / num_points * 3)))

    # Acumuladores de probabilidad
    prob_sum = np.zeros((N, 2), dtype=np.float32)
    prob_count = np.zeros(N, dtype=np.int32)

    model.eval()
    with torch.no_grad():
        done = 0
        while done < n_passes:
            # Batch de subsamples para eficiencia GPU
            b_size = min(_INFER_BATCH, n_passes - done)
            b_xyz, b_feat, b_idx = [], [], []

            for _ in range(b_size):
                # Submuestreo global (replica ForInstanceDataset.__getitem__)
                if N >= num_points:
                    choice = rng.choice(N, num_points, replace=False)
                else:
                    choice = rng.choice(N, num_points, replace=True)

                # Normalización per-sample (replica __getitem__)
                xyz_s = xyz_c[choice].copy()
                centroid = xyz_s.mean(axis=0)
                xyz_s -= centroid
                radius = float(np.max(np.sqrt((xyz_s ** 2).sum(axis=1)))) + 1e-8
                xyz_s /= radius

                hag_s = hag_norm_all[choice]
                int_s = intensity_all[choice]

                b_xyz.append(xyz_s)
                b_feat.append(np.column_stack([hag_s, int_s]).astype(np.float32))
                b_idx.append(choice)

            xyz_t = torch.from_numpy(np.stack(b_xyz)).to(device)   # (B, N, 3)
            feat_t = torch.from_numpy(np.stack(b_feat)).to(device)  # (B, N, 2)

            with autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=cfg.pointnet2.mixed_precision,
            ):
                logits = model(xyz_t, feat_t)   # (B, num_points, 2) log-probs

            probs_batch = torch.exp(logits).float().cpu().numpy()   # (B, num_points, 2)

            for indices, probs in zip(b_idx, probs_batch):
                np.add.at(prob_sum, indices, probs)
                np.add.at(prob_count, indices, 1)

            done += b_size

    # Normalizar probabilidades de puntos cubiertos
    covered_mask = prob_count > 0
    pred_proba = np.zeros((N, 2), dtype=np.float32)
    pred_proba[covered_mask] = prob_sum[covered_mask] / prob_count[covered_mask, None]

    # Diagnóstico de cobertura
    cov = compute_coverage(prob_count)
    print(f"    Coverage: {cov['covered_points']}/{cov['total_points']} "
          f"({cov['coverage_pct']:.1f}%) — "
          f"{cov['uncovered_count']} puntos sin cubrir")

    # Puntos no cubiertos: KNN ponderado desde cubiertos más cercanos
    if cov["uncovered_count"] > 0:
        pred_proba = assign_uncovered_points(xyz_c, pred_proba, covered_mask, k=3)
        print(f"    KNN interpolation aplicado a {cov['uncovered_count']} puntos.")

    pred_binary = pred_proba.argmax(axis=1).astype(np.int32)

    return pred_binary, pred_proba


def load_model(cfg, device: torch.device) -> PointNet2SemSegMSG:
    """
    Carga el mejor modelo entrenado desde disco.

    Args:
        cfg:    OmegaConf config.
        device: torch.device.

    Returns:
        Modelo en eval mode.

    Raises:
        FileNotFoundError: Si el modelo no existe.
    """
    model_path = (
        Path(cfg.paths.output_dir) / "models" /
        cfg.pointnet2.model_save_path.split("/")[-1]
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo PointNet++ no encontrado: {model_path}\n"
            "Entrena primero con: python -m forest_its.methods.pointnet2.train_pointnet2"
        )
    model = PointNet2SemSegMSG(num_classes=cfg.pointnet2.num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
