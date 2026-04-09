"""
Extracción de 28 features geométricos para clasificación semántica con Random Forest.

Features basados en Weinmann et al. (2017) "Geometric Features and their Relevance
for 3D Point Cloud Classification" (ISPRS Annals). Los 8 features eigen-based
se calculan a partir de la matriz de covarianza del vecindario KNN, y se
complementan con features de altura, densidad y retorno LiDAR.

Se calculan en DOS escalas (k=20 y k=50 vecinos), dando 28 features por punto.
La estrategia multi-escala sigue Weinmann et al. (2015) y permite capturar
tanto detalle fino (ramas, troncos) como contexto grueso (posición en el rodal).

Convención de normalización de eigenvalores:
  - Features 1-6 y 8: usan eigenvalores normalizados λi_norm = λi / (λ1+λ2+λ3)
  - Feature 7 (suma): usa eigenvalores crudos λ1+λ2+λ3
  Esto sigue Weinmann et al. (2013, 2014, 2015).

Covarianza se calcula con divisor k (no k-1), consistente con la literatura
de point cloud processing (Weinmann 2014, 2015).

KNN usa scipy.spatial.cKDTree para portabilidad y estabilidad.
Para datasets dispersos (RMIT ~454 pts/m²), se aplica un radio máximo de
búsqueda: si el k-ésimo vecino está a más de max_neighbor_distance metros,
se usan solo los vecinos dentro de ese radio.
"""

import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
from tqdm import tqdm


# Nombres de features para reportes y análisis de importancia
FEATURE_NAMES_BASE = [
    "linearity", "planarity", "sphericity", "omnivariance",
    "anisotropy", "eigenentropy", "sum_eigenvalues", "change_curvature",
    "HAG", "verticality", "density", "roughness", "height_range", "intensity",
]

FEATURE_NAMES_28 = (
    [f"{name}_k20" for name in FEATURE_NAMES_BASE]
    + [f"{name}_k50" for name in FEATURE_NAMES_BASE]
)


def compute_eigenfeatures(xyz_neighbors: np.ndarray) -> tuple:
    """
    Calcula los 8 features eigen-based para un vecindario dado.

    Implementación:
      1. Centrar: xyz_centered = xyz_neighbors - mean
      2. Covarianza: C = (xyz_centered.T @ xyz_centered) / k
         NOTA: divide por k, no por k-1. Consistente con Weinmann et al. (2014).
      3. Eigenvalores: λ1 >= λ2 >= λ3 via np.linalg.eigh (más estable que eig)
         eigh retorna en orden ascendente -> se invierte
      4. Normalizar: λi_norm = λi / (λ1+λ2+λ3 + eps)
      5. Manejar degenerate cases: si Σλ < eps, retornar zeros

    Args:
        xyz_neighbors: (k, 3) coordenadas de los k vecinos.

    Returns:
        eigenfeats: (8,) float32 con los 8 features.
        normal: (3,) eigenvector del menor eigenvalue (normal local).
    """
    k = xyz_neighbors.shape[0]
    eps = 1e-10

    # Centrar
    centroid = xyz_neighbors.mean(axis=0)
    centered = xyz_neighbors - centroid

    # Covarianza con divisor k (no k-1)
    cov = (centered.T @ centered) / k

    # Eigendecomposición
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Invertir para λ1 >= λ2 >= λ3
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Normal local = eigenvector del menor eigenvalue (columna 2 tras invertir)
    normal = eigenvectors[:, 2]

    # Clamp negativos
    eigenvalues = np.maximum(eigenvalues, eps)
    lam1, lam2, lam3 = eigenvalues
    lam_sum = lam1 + lam2 + lam3

    if lam_sum < eps:
        return np.zeros(8, dtype=np.float32), normal

    l1n = lam1 / lam_sum
    l2n = lam2 / lam_sum
    l3n = lam3 / lam_sum

    linearity = (lam1 - lam2) / lam1
    planarity = (lam2 - lam3) / lam1
    sphericity = lam3 / lam1
    omnivariance = (lam1 * lam2 * lam3) ** (1.0 / 3.0)
    anisotropy = (lam1 - lam3) / lam1
    eigen_entropy = -(l1n * np.log(l1n + eps)
                      + l2n * np.log(l2n + eps)
                      + l3n * np.log(l3n + eps))
    sum_eigenvalues = lam_sum  # CRUDOS, no normalizados
    change_of_curvature = lam3 / lam_sum

    feats = np.array([
        linearity, planarity, sphericity, omnivariance,
        anisotropy, eigen_entropy, sum_eigenvalues, change_of_curvature,
    ], dtype=np.float32)

    return feats, normal


def compute_local_features(
    point_hag: float,
    point_intensity: float,
    normal: np.ndarray,
    neighbors_xyz: np.ndarray,
    point_xyz: np.ndarray,
    density_radius: float = 0.5,
) -> np.ndarray:
    """
    Calcula los 6 features restantes (9-14) para un punto dado sus vecinos.

    Features:
      9.  HAG — feature #1 según Bremer et al. (2023), peso=0.21
      10. Verticalidad: 1 - |dot(normal, [0,0,1])| — ángulo vs vertical
      11. Densidad local: count de vecinos en radio fijo density_radius
      12. Rugosidad: std(Z de vecinos)
      13. Rango de altura: max(Z) - min(Z) en vecindario
      14. Intensidad normalizada [0,1]

    Args:
        point_hag: HAG del punto central.
        point_intensity: Intensidad normalizada.
        normal: (3,) normal local.
        neighbors_xyz: (k, 3) coordenadas de vecinos.
        point_xyz: (3,) coordenadas del punto.
        density_radius: Radio para densidad (metros).

    Returns:
        (6,) float32.
    """
    verticality = 1.0 - abs(float(normal[2]))
    dists = np.linalg.norm(neighbors_xyz - point_xyz, axis=1)
    density = float(np.sum(dists <= density_radius))
    roughness = float(neighbors_xyz[:, 2].std())
    height_range = float(neighbors_xyz[:, 2].max() - neighbors_xyz[:, 2].min())

    return np.array([
        point_hag, verticality, density, roughness, height_range, point_intensity,
    ], dtype=np.float32)


def compute_features_batch(
    xyz: np.ndarray,
    hag: np.ndarray,
    intensity: np.ndarray,
    k_small: int = 20,
    k_large: int = 50,
    batch_size: int = 10000,
    density_radius: float = 0.5,
    max_neighbor_distance: float = 5.0,
) -> np.ndarray:
    """
    Calcula 28 features para TODOS los puntos de una nube.

    Proceso:
      1. Construir KDTree global con scipy.spatial.cKDTree
      2. Pre-query todos los k_large+1 vecinos de una vez
      3. Para cada punto, extraer features a ambas escalas
      4. Radio máximo: vecinos más allá de max_neighbor_distance se descartan

    Args:
        xyz: (N, 3) float64.
        hag: (N,) float32.
        intensity: (N,) float32.
        k_small: Escala fina (20).
        k_large: Escala gruesa (50).
        batch_size: Para barra de progreso.
        density_radius: Radio densidad local (0.5m).
        max_neighbor_distance: Radio máximo KNN (5.0m).

    Returns:
        features: (N, 28) float32.
    """
    n_points = xyz.shape[0]
    features = np.zeros((n_points, 28), dtype=np.float32)

    # KDTree con scipy
    tree = cKDTree(xyz)

    # Pre-query todos los vecinos a escala grande (k_large+1 incluye self)
    max_k = k_large + 1
    distances, indices = tree.query(xyz, k=max_k)

    n_batches = (n_points + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc="Computing RF features"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_points)

        for i in range(start, end):
            for scale_idx, k in enumerate([k_small, k_large]):
                # Vecinos excluyendo self (índice 0)
                nbr_idx = indices[i, 1:k + 1]
                nbr_dists = distances[i, 1:k + 1]

                # Filtrar por radio máximo
                valid = (nbr_idx < n_points) & (nbr_dists <= max_neighbor_distance)
                nbr_idx = nbr_idx[valid]

                if len(nbr_idx) < 4:
                    continue

                neighbors_xyz = xyz[nbr_idx]

                # Eigen features (1-8)
                eigen_feats, normal = compute_eigenfeatures(neighbors_xyz)

                # Local features (9-14)
                local_feats = compute_local_features(
                    hag[i], intensity[i], normal,
                    neighbors_xyz, xyz[i],
                    density_radius=density_radius,
                )

                features[i, scale_idx * 14: scale_idx * 14 + 8] = eigen_feats
                features[i, scale_idx * 14 + 8: (scale_idx + 1) * 14] = local_feats

    return features


def compute_features_for_plot(
    las_data: dict,
    cfg,
    output_dir: Path,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Wrapper completo: normaliza, extrae features, gestiona cache.

    Cache: guarda features en output_dir/features_cache/{plot_stem}_features.npy.
    Si el cache existe y force_recompute=False, carga directamente.

    ADVERTENCIA DE RENDIMIENTO: con ~18M puntos totales puede tardar 30-90 min.
    El cache es CRÍTICO para desarrollo iterativo.

    Args:
        las_data: Dict de load_las() con campo 'hag' ya calculado.
        cfg: Configuración.
        output_dir: Directorio base.
        force_recompute: Forzar recálculo.

    Returns:
        features: (N_valid, 28) float32.
    """
    from forest_its.data.dataset import get_binary_labels

    plot_stem = las_data.get("_plot_stem", "unknown")
    cache_dir = Path(output_dir) / cfg.features.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{plot_stem}_features.npy"

    binary_labels = get_binary_labels(las_data["classification"])
    valid_mask = binary_labels != -1

    if cache_path.exists() and not force_recompute:
        features = np.load(cache_path)
        if features.shape[0] == valid_mask.sum():
            print(f"  Loaded cached features: {cache_path}")
            return features
        print(f"  Cache size mismatch, recomputing...")

    xyz_valid = las_data["xyz"][valid_mask]
    hag_valid = las_data["hag"][valid_mask]
    intensity_valid = las_data["intensity"][valid_mask]

    features = compute_features_batch(
        xyz_valid,
        hag_valid,
        intensity_valid,
        k_small=cfg.features.k_small,
        k_large=cfg.features.k_large,
        batch_size=cfg.features.batch_size,
        density_radius=cfg.features.density_radius,
        max_neighbor_distance=cfg.features.max_neighbor_distance,
    )

    np.save(cache_path, features)
    print(f"  Cached features: {cache_path}")

    return features
