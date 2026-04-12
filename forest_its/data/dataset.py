"""
Loader del dataset FOR-instance (Puliti et al., 2023).

FOR-instance es un benchmark de UAV laser scanning con anotaciones semánticas
y de instancia por punto, diseñado para evaluar métodos de segmentación de
árboles individuales (ITS).

Clases semánticas (del README oficial):
    0 = Unclassified  -> excluir
    1 = Low-vegetation -> no-árbol
    2 = Terrain        -> no-árbol
    3 = Out-points     -> excluir (árboles fuera del plot anotado)
    4 = Stem           -> árbol
    5 = Live-branches  -> árbol
    6 = Woody-branches -> árbol
"""

import numpy as np
import pandas as pd
import laspy
from pathlib import Path
from typing import Dict, List, Tuple


# Colecciones excluidas explícitamente del estudio (paper §3.1). NIBIO2
# no forma parte de FOR-instance V1 y se descarta aunque aparezca tanto
# en el CSV oficial como en disco.
EXCLUDE_FOLDERS = frozenset({"NIBIO2"})


def load_las(path: Path) -> Dict[str, np.ndarray]:
    """
    Carga un archivo .las de FOR-instance.

    Usa laspy.read() para acceder a todos los campos estándar y extra del
    formato LAS. Los campos 'Classification' y 'treeID' son anotaciones
    oficiales del dataset provistas como dimensiones extra.

    Args:
        path: Ruta absoluta al archivo .las.

    Returns:
        Diccionario con:
          - xyz: (N, 3) float64, coordenadas 3D
          - intensity: (N,) float32, normalizada a [0, 1]
          - return_number: (N,) int32
          - number_of_returns: (N,) int32
          - classification: (N,) int32, etiqueta semántica del dataset
          - tree_id: (N,) int32, ID de instancia por árbol.
            treeID == 0 significa "sin árbol asignado" (suelo, vegetación baja,
            o puntos no anotados). NO es un árbol con ID 0.
    """
    las = laspy.read(str(path))

    xyz = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)

    # Normalizar intensidad a [0, 1]
    raw_intensity = np.array(las.intensity, dtype=np.float32)
    imax = raw_intensity.max()
    intensity = raw_intensity / imax if imax > 0 else raw_intensity

    return_number = np.array(las.return_number, dtype=np.int32)
    number_of_returns = np.array(las.number_of_returns, dtype=np.int32)

    # Campos de anotación del dataset FOR-instance
    classification = np.array(las.classification, dtype=np.int32)
    tree_id = np.array(las.treeID, dtype=np.int32)

    return {
        "xyz": xyz,
        "intensity": intensity,
        "return_number": return_number,
        "number_of_returns": number_of_returns,
        "classification": classification,
        "tree_id": tree_id,
    }


def get_binary_labels(classification: np.ndarray) -> np.ndarray:
    """
    Convierte etiquetas semánticas de FOR-instance a binario árbol/no-árbol.

    Mapeo definido en el README oficial del dataset (Puliti et al., 2023):
      - árbol    (1): Classification in {4, 5, 6} (Stem, Live-branches, Woody-branches)
      - no-árbol (0): Classification in {1, 2}    (Low-vegetation, Terrain)
      - excluir (-1): Classification in {0, 3}    (Unclassified, Out-points)

    Los puntos con label == -1 deben enmascararse completamente: no participan
    en entrenamiento, validación ni evaluación. Incluirlos introduciría ruido
    semántico sin ground truth confiable. Los Out-points (clase 3) corresponden
    a árboles fuera del plot de medición, cuya anotación de instancia es
    incompleta y sesgaría las métricas.

    Args:
        classification: (N,) array de etiquetas semánticas originales.

    Returns:
        (N,) array int32 con valores {-1, 0, 1}.
    """
    labels = np.full(classification.shape, -1, dtype=np.int32)
    labels[np.isin(classification, [1, 2])] = 0   # no-árbol
    labels[np.isin(classification, [4, 5, 6])] = 1  # árbol
    return labels


def get_all_plots(dataset_root: Path) -> List[Path]:
    """
    Retorna lista de paths de todos los archivos .las en el dataset.

    Busca recursivamente en todas las subcarpetas institucionales.

    Args:
        dataset_root: Ruta raíz del dataset FOR-instance.

    Returns:
        Lista de Path a cada archivo .las, ordenada alfabéticamente.
    """
    plots = sorted(dataset_root.rglob("*.las"))
    return plots


def load_splits(dataset_root: Path) -> Tuple[List[Path], List[Path]]:
    """
    Lee data_split_metadata.csv y retorna las listas de archivos dev y test.

    El CSV oficial tiene columnas: path, folder, split.
    'split' es 'dev' o 'test'. 'path' es relativo a dataset_root.

    Filtra automáticamente:
      - Colecciones en `EXCLUDE_FOLDERS` (actualmente NIBIO2, excluida
        explícitamente por protocolo del paper §3.1).
      - Archivos que no existen en disco (p. ej. colecciones no
        descargadas en una copia parcial del dataset).

    El split oficial es necesario para comparabilidad con ForAINet
    (Henrich et al., 2024) y SegmentAnyTree (Wielgosz et al., 2024),
    que usan el mismo benchmark y partición.

    Args:
        dataset_root: Ruta raíz del dataset FOR-instance.

    Returns:
        (dev_paths, test_paths): tupla de listas de Path absolutos
        (solo archivos que existen en disco).
    """
    csv_path = dataset_root / "data_split_metadata.csv"
    df = pd.read_csv(csv_path)

    dev_paths = []
    test_paths = []
    skipped_missing = 0
    skipped_excluded = 0

    for _, row in df.iterrows():
        folder = str(row.get("folder", "")).strip()
        if folder in EXCLUDE_FOLDERS:
            skipped_excluded += 1
            continue

        file_path = dataset_root / row["path"]
        if not file_path.exists():
            skipped_missing += 1
            continue
        if row["split"] == "dev":
            dev_paths.append(file_path)
        elif row["split"] == "test":
            test_paths.append(file_path)

    if skipped_excluded > 0:
        print(
            f"  [INFO] Excluded {skipped_excluded} files from folders "
            f"{sorted(EXCLUDE_FOLDERS)} by protocol (paper §3.1)"
        )
    if skipped_missing > 0:
        print(
            f"  [INFO] Skipped {skipped_missing} files from CSV not found on disk"
        )

    return dev_paths, test_paths
