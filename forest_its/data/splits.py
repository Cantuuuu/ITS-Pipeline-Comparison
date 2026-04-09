"""
Lógica de partición train/val usando el split oficial de FOR-instance.

El split dev/test viene definido por data_split_metadata.csv (Puliti et al., 2023).
Dentro de dev, se realiza una partición adicional train/val a nivel de plot
(archivo .las) para evitar data leakage espacial.

Justificación del split por plot: los puntos dentro de un mismo plot están
espacialmente correlacionados. Si se partiera a nivel de punto, puntos del
mismo árbol podrían caer en train y val simultáneamente, inflando artificialmente
las métricas de validación. Este criterio es consistente con ForAINet
(Henrich et al., 2024) y SegmentAnyTree (Wielgosz et al., 2024).
"""

from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split


def get_train_val_split(
    dev_paths: List[Path],
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """
    Divide dev_paths en conjuntos train y val a nivel de plot.

    El split se realiza a nivel de archivo .las (plot), no a nivel de punto,
    para evitar data leakage espacial entre train y val.

    Args:
        dev_paths: Lista de paths a archivos .las del conjunto dev.
        val_fraction: Fracción de plots para validación (default 0.2).
        random_state: Semilla para reproducibilidad (default 42).

    Returns:
        (train_paths, val_paths): tupla de listas de Path.
    """
    train_paths, val_paths = train_test_split(
        dev_paths,
        test_size=val_fraction,
        random_state=random_state,
    )
    return train_paths, val_paths
