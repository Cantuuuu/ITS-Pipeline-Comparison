"""
Selección del backend de cómputo para PointNet++.

Prioridad: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU.

El repo se desarrolla y entrena principalmente en Apple Silicon (M4 Pro), por
lo que MPS tiene preferencia sobre CUDA cuando ambos están disponibles. Si en
una máquina solo está CUDA, se usa CUDA. En su defecto, CPU.
"""

import torch


def select_device() -> torch.device:
    """Devuelve el mejor torch.device disponible."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
