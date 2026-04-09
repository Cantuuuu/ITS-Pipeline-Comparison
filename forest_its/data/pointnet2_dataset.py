"""
Dataset PyTorch para entrenamiento y validación de PointNet++ sobre FOR-instance.
"""

import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from forest_its.data.dataset import load_las, get_binary_labels
from forest_its.preprocessing.normalize_height import process_plot


class ForInstanceDataset(Dataset):
    """
    Dataset para PointNet++ sobre FOR-instance.

    Estrategia de muestreo (num_points=8192 por muestra):
      - Si el plot tiene > num_points puntos válidos (label != -1):
          Submuestreo aleatorio en cada __getitem__ (data augmentation implícita)
      - Si el plot tiene < num_points puntos válidos:
          Oversampling con reemplazo hasta completar num_points

    Augmentation en training:
      - Jitter aleatorio en XYZ: Gaussiano con std=jitter_std (simula ruido GPS)
      - Flip aleatorio en X: simula orientación del vuelo
      - Scale aleatorio [scale_min, scale_max]: simula variación de altura de vuelo
      NO aplicar rotación en Z — el dataset tiene orientaciones consistentes

    Normalización (por bloque, en __getitem__):
      - XYZ: centrar en centroide del bloque, dividir por radio máximo
      - HAG: normalizar a [0, 1] usando hag_max (cfg.preprocessing.hag_max = 50m)
      - Intensity: ya normalizada [0,1] en load_las()

    Retorna:
      points: (num_points, 5) float32 — [X_norm, Y_norm, Z_norm, HAG_norm, intensity]
      labels: (num_points,) int64 — {0: no-árbol, 1: árbol}
      mask:   (num_points,) bool — True si el punto es válido (label != -1).
              Siempre True en training: los puntos -1 se excluyen en __init__.
    """

    def __init__(
        self,
        las_paths: list,
        num_points: int = 8192,
        augment: bool = True,
        cfg=None,
    ):
        """
        Args:
            las_paths:  Lista de paths a archivos .las/.laz.
            num_points: Puntos por muestra.
            augment:    Aplicar data augmentation (solo en train).
            cfg:        OmegaConf config. Necesario para parámetros de preprocesamiento.
        """
        self.num_points = num_points
        self.augment = augment

        # Parámetros de normalización/augmentation
        self.hag_max = float(cfg.preprocessing.hag_max) if cfg else 50.0
        self.jitter_std = float(cfg.pointnet2.jitter_std) if cfg else 0.02
        self.scale_min = float(cfg.pointnet2.scale_min) if cfg else 0.9
        self.scale_max = float(cfg.pointnet2.scale_max) if cfg else 1.1

        self.plots = []
        for las_path in las_paths:
            data = load_las(las_path)
            binary_labels = get_binary_labels(data["classification"])

            # Computar HAG usando el mismo pipeline que RF (normalización consistente)
            if cfg is not None:
                process_plot(
                    data,
                    resolution_dtm=cfg.preprocessing.dtm_resolution,
                    smooth_window=cfg.preprocessing.smooth_window,
                    hag_min=cfg.preprocessing.hag_min,
                    hag_max=cfg.preprocessing.hag_max,
                )
                hag_all = data["hag"].astype(np.float32)
            else:
                hag_all = np.zeros(len(data["xyz"]), dtype=np.float32)

            # Solo puntos con label válido (excluir clases 0 y 3 → label == -1)
            valid = binary_labels != -1
            xyz = data["xyz"][valid].astype(np.float32)
            hag = hag_all[valid]
            intensity = data["intensity"][valid].astype(np.float32)
            labels = binary_labels[valid].astype(np.int64)

            # Pre-centrar XY en centroide del plot (reduce error numérico en float32)
            xy_center = xyz[:, :2].mean(axis=0)
            xyz[:, 0] -= xy_center[0]
            xyz[:, 1] -= xy_center[1]

            # Normalizar HAG a [0, 1]
            hag_norm = np.clip(hag / self.hag_max, 0.0, 1.0)

            self.plots.append({
                "xyz": xyz,           # (N, 3) float32, XY centrado en plot
                "hag_norm": hag_norm,  # (N,)  float32, [0,1]
                "intensity": intensity,  # (N,) float32, [0,1]
                "labels": labels,      # (N,)  int64
                "path": str(las_path),
            })

    def __len__(self):
        # Número de plots, NO de puntos.
        # En training se submuestrea aleatoriamente en __getitem__.
        return len(self.plots)

    def __getitem__(self, idx):
        plot = self.plots[idx]
        n = len(plot["labels"])

        # --- Submuestreo ---
        if n >= self.num_points:
            choice = np.random.choice(n, self.num_points, replace=False)
        else:
            # Oversampling con reemplazo si el plot es pequeño
            choice = np.random.choice(n, self.num_points, replace=True)

        xyz = plot["xyz"][choice].copy()         # (num_points, 3)
        hag_norm = plot["hag_norm"][choice].copy()  # (num_points,)
        intensity = plot["intensity"][choice].copy()  # (num_points,)
        labels = plot["labels"][choice]              # (num_points,)

        # --- Normalización por bloque ---
        # Centrar en centroide del bloque seleccionado
        centroid = xyz.mean(axis=0)
        xyz -= centroid
        # Normalizar por radio máximo del bloque
        radius = np.max(np.sqrt(np.sum(xyz ** 2, axis=1))) + 1e-8
        xyz /= radius

        # --- Augmentation (solo en train) ---
        if self.augment:
            # Jitter XYZ (simula ruido de posicionamiento GPS, std en espacio normalizado)
            xyz += np.random.normal(0, self.jitter_std, xyz.shape).astype(np.float32)
            # Flip aleatorio en X (simula orientación del vuelo)
            if np.random.random() > 0.5:
                xyz[:, 0] = -xyz[:, 0]
            # Scale aleatorio (simula variación de altura de vuelo)
            scale = np.random.uniform(self.scale_min, self.scale_max)
            xyz *= scale

        # --- Construir tensor de features (num_points, 5) ---
        # [X_norm, Y_norm, Z_norm, HAG_norm, intensity]
        points = np.column_stack([
            xyz,
            hag_norm[:, None],
            intensity[:, None],
        ]).astype(np.float32)

        # mask: todos True (los puntos inválidos se excluyeron en __init__)
        mask = np.ones(self.num_points, dtype=bool)

        return (
            torch.from_numpy(points),   # (num_points, 5)
            torch.from_numpy(labels),   # (num_points,)
            torch.from_numpy(mask),     # (num_points,) bool
        )
