"""
Extracción de 27 features geométricos para clasificación semántica con Random Forest.

Features basados en Weinmann et al. (2017) "Geometric Features and their Relevance
for 3D Point Cloud Classification" (ISPRS Annals). Los 8 features eigen-based
se calculan a partir de la matriz de covarianza del vecindario KNN, y se
complementan con features de altura, rugosidad y retorno LiDAR.

Se computan 13 features dependientes de escala en DOS escalas (k=20 y k=50
vecinos) más 1 feature de densidad volumétrica compartida entre escalas
(cardinalidad real de la bola de radio fijo `density_radius`, calculada
una sola vez por punto vía `cKDTree.query_ball_point`), dando 27 features
por punto. La estrategia multi-escala sigue Weinmann et al. (2015) y
permite capturar tanto detalle fino (ramas, troncos) como contexto grueso
(posición en el rodal).

Convención de normalización de eigenvalores:
  - Features 1-6 y 8 usan eigenvalores normalizados λi_norm = λi / (λ1+λ2+λ3).
    En particular, omnivarianza se computa como (λ1n * λ2n * λ3n)^(1/3) en
    el espacio normalizado, siguiendo Weinmann et al. (2013, 2014, 2015).
  - Feature 7 (suma) es el único descriptor que usa eigenvalores crudos
    λ1+λ2+λ3 (traza de la matriz de covarianza, no una distribución).
  Las features derivadas como ratios (linealidad, planaridad, esfericidad,
  anisotropía, cambio de curvatura) son invariantes a la normalización y
  coinciden en ambas convenciones.

Covarianza se calcula con divisor k (no k-1), consistente con la literatura
de point cloud processing (Weinmann 2014, 2015).

KNN usa scipy.spatial.cKDTree para portabilidad y estabilidad.
Para los plots de menor densidad del benchmark (ver §3.1 del paper
para el rango medido sobre las cinco colecciones), se aplica un radio
máximo de búsqueda: si el k-ésimo vecino está a más de
max_neighbor_distance metros, se usan solo los vecinos dentro de ese
radio.
"""

import os
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed


# Nombres de features para reportes y análisis de importancia.
# Los 13 features de FEATURE_NAMES_BASE se computan a dos escalas (k=20, k=50);
# `density` (cardinalidad volumétrica real) se computa una sola vez por punto
# vía cKDTree.query_ball_point y se comparte entre escalas.
FEATURE_NAMES_BASE = [
    "linearity", "planarity", "sphericity", "omnivariance",
    "anisotropy", "eigenentropy", "sum_eigenvalues", "change_curvature",
    "HAG", "verticality", "roughness", "height_range", "intensity",
]

FEATURE_NAMES_27 = (
    [f"{name}_k20" for name in FEATURE_NAMES_BASE]
    + [f"{name}_k50" for name in FEATURE_NAMES_BASE]
    + ["density"]
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
    # Omnivarianza en el espacio normalizado (Weinmann et al. 2015).
    omnivariance = (l1n * l2n * l3n) ** (1.0 / 3.0)
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
) -> np.ndarray:
    """
    Calcula los 5 features locales dependientes de escala (no eigen-based).

    Features:
      9.  HAG — feature #1 según Bremer et al. (2023), peso=0.21
      10. Verticalidad: 1 - |dot(normal, [0,0,1])| — ángulo vs vertical
      11. Rugosidad: std(Z de vecinos)
      12. Rango de altura: max(Z) - min(Z) en vecindario
      13. Intensidad normalizada [0,1]

    La densidad volumétrica (cardinalidad de la bola de radio fijo
    `density_radius`) ya no se computa aquí: se calcula una sola vez por
    punto en `compute_features_batch` vía `cKDTree.query_ball_point` y
    se concatena al final como feature compartida entre escalas.

    Args:
        point_hag: HAG del punto central.
        point_intensity: Intensidad normalizada.
        normal: (3,) normal local.
        neighbors_xyz: (k, 3) coordenadas de vecinos.

    Returns:
        (5,) float32.
    """
    verticality = 1.0 - abs(float(normal[2]))
    roughness = float(neighbors_xyz[:, 2].std())
    height_range = float(neighbors_xyz[:, 2].max() - neighbors_xyz[:, 2].min())

    return np.array([
        point_hag, verticality, roughness, height_range, point_intensity,
    ], dtype=np.float32)


def _features_chunk(
    chunk_indices: np.ndarray,
    xyz: np.ndarray,
    hag: np.ndarray,
    intensity: np.ndarray,
    density_chunk: np.ndarray,
    distances_chunk: np.ndarray,
    indices_chunk: np.ndarray,
    k_small: int,
    k_large: int,
    max_neighbor_distance: float,
) -> np.ndarray:
    """
    Procesa un sub-rango de puntos. Función pura: no muta inputs.
    Devuelve (len(chunk_indices), 27).

    Layout de las 27 columnas:
      [0:8]   eigen features k20    [8:13]  local features k20
      [13:21] eigen features k50    [21:26] local features k50
      [26]    density (compartida entre escalas)

    Diseñada para ejecutarse en workers de joblib (loky). Recibe slices ya
    extraídos de `distances`, `indices` y `density_per_point` para evitar
    pasar los arrays completos por IPC.
    """
    n_points = xyz.shape[0]
    n_chunk = len(chunk_indices)
    out = np.zeros((n_chunk, 27), dtype=np.float32)

    for local_i, i in enumerate(chunk_indices):
        for scale_idx, k in enumerate([k_small, k_large]):
            nbr_idx = indices_chunk[local_i, 1:k + 1]
            nbr_dists = distances_chunk[local_i, 1:k + 1]

            valid = (nbr_idx < n_points) & (nbr_dists <= max_neighbor_distance)
            nbr_idx = nbr_idx[valid]

            if len(nbr_idx) < 4:
                continue

            neighbors_xyz = xyz[nbr_idx]

            eigen_feats, normal = compute_eigenfeatures(neighbors_xyz)
            local_feats = compute_local_features(
                hag[i], intensity[i], normal, neighbors_xyz,
            )

            out[local_i, scale_idx * 13: scale_idx * 13 + 8] = eigen_feats
            out[local_i, scale_idx * 13 + 8: (scale_idx + 1) * 13] = local_feats

    out[:, 26] = density_chunk
    return out


def compute_features_batch(
    xyz: np.ndarray,
    hag: np.ndarray,
    intensity: np.ndarray,
    k_small: int = 20,
    k_large: int = 50,
    batch_size: int = 10000,
    density_radius: float = 0.5,
    max_neighbor_distance: float = 5.0,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Calcula 27 features para TODOS los puntos de una nube.

    Proceso:
      1. Construir KDTree global con scipy.spatial.cKDTree
      2. Pre-query todos los k_large+1 vecinos en paralelo (workers=-1)
      3. Pre-computar densidad volumétrica real (cardinalidad de la bola
         de radio density_radius) una sola vez por punto vía
         tree.query_ball_point(..., return_length=True). Esta es la
         feature #27, compartida entre escalas.
      4. Particionar los puntos en chunks y delegar a workers loky
      5. Radio máximo: vecinos más allá de max_neighbor_distance se descartan

    Args:
        xyz: (N, 3) float64.
        hag: (N,) float32.
        intensity: (N,) float32.
        k_small: Escala fina (20).
        k_large: Escala gruesa (50).
        batch_size: Tamaño nominal de chunk para joblib (no afecta correctness).
        density_radius: Radio para la densidad volumétrica (0.5m).
        max_neighbor_distance: Radio máximo KNN (5.0m).
        n_jobs: Workers paralelos (-1 = todos los cores).

    Returns:
        features: (N, 27) float32.
    """
    n_points = xyz.shape[0]

    # KDTree con scipy + query paralela (scipy >= 1.6 soporta workers=-1)
    tree = cKDTree(xyz)
    max_k = k_large + 1
    distances, indices = tree.query(xyz, k=max_k, workers=n_jobs)

    # Densidad volumétrica real (cardinalidad de la bola de radio fijo).
    # Una sola pasada por punto, sin acotamiento por k_large.
    density_per_point = np.asarray(
        tree.query_ball_point(xyz, r=density_radius, return_length=True),
        dtype=np.float32,
    )

    # Decidir tamaño de chunk: balance entre overhead IPC y granularidad
    # Por defecto ~1 chunk por core, mínimo `batch_size` puntos por chunk.
    n_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    chunk_size = max(batch_size, (n_points + n_workers - 1) // n_workers)
    chunk_starts = list(range(0, n_points, chunk_size))

    # Cada chunk recibe slices ya extraídos para minimizar serialización
    chunks = [
        (
            np.arange(s, min(s + chunk_size, n_points)),
            density_per_point[s:min(s + chunk_size, n_points)],
            distances[s:min(s + chunk_size, n_points)],
            indices[s:min(s + chunk_size, n_points)],
        )
        for s in chunk_starts
    ]

    print(
        f"  RF features: {n_points:,} pts, {len(chunks)} chunks, "
        f"n_jobs={n_workers}"
    )

    results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
        delayed(_features_chunk)(
            ch_idx, xyz, hag, intensity, ch_dens, ch_dist, ch_inds,
            k_small, k_large, max_neighbor_distance,
        )
        for ch_idx, ch_dens, ch_dist, ch_inds in tqdm(chunks, desc="Computing RF features")
    )

    features = np.concatenate(results, axis=0).astype(np.float32)
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
        features: (N_valid, 27) float32.
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
        if features.shape[0] == valid_mask.sum() and features.shape[1] == 27:
            print(f"  Loaded cached features: {cache_path}")
            return features
        print(f"  Cache mismatch (expected (N_valid, 27)), recomputing...")

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
