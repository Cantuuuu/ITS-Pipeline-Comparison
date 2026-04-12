"""
Entrenamiento PointNet++ MSG para segmentación semántica binaria.

Uso:
  python -m forest_its.methods.pointnet2.train_pointnet2 --dry-run   # smoke test SOLO
  python -m forest_its.methods.pointnet2.train_pointnet2              # entrenamiento real (MAÑANA)

--dry-run:
  - Carga 1 plot de train
  - Construye el modelo
  - Hace 1 forward pass con batch_size=1
  - Calcula loss
  - Hace 1 backward pass
  - Imprime shapes de todos los tensores intermedios
  - Verifica que el modelo cabe en memoria del dispositivo activo
  - NO guarda nada, NO itera epochs
  - Tiempo esperado: < 60 segundos
"""

import sys
import time
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm


def set_seed(seed: int):
    """Fija semillas globales para reproducibilidad (paper §3.3, Apéndice A)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from forest_its.data.dataset import load_splits
from forest_its.data.splits import get_train_val_split
from forest_its.data.pointnet2_dataset import ForInstanceDataset
from forest_its.methods.pointnet2.model_msg import PointNet2SemSegMSG
from forest_its.methods.pointnet2.device_utils import select_device


# bf16 (sin GradScaler) en MPS y CUDA modernas; fp16 en CUDA antiguas no se usa
# en este repo. La función `_amp_dtype` centraliza la elección para que las
# llamadas a autocast queden uniformes.
def _amp_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16


# ─────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────

def _worker_init_fn(worker_id: int):
    """Seed numpy por worker para que augmentation no se duplique entre workers."""
    np.random.seed(42 + worker_id)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_class_weights(dataset: ForInstanceDataset) -> torch.Tensor:
    """
    Calcula pesos de clase inversamente proporcionales a su frecuencia.

    Compensa el desbalance árbol/no-árbol típico en UAV LiDAR forestal.
    """
    all_labels = np.concatenate([p["labels"] for p in dataset.plots])
    counts = np.bincount(all_labels, minlength=2)
    total = counts.sum()
    weights = total / (2.0 * counts + 1e-6)
    return torch.FloatTensor(weights)


def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> float:
    """mIoU sobre batch. pred y target shape: (B*N,)."""
    ious = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float("nan"))
    valid = [x for x in ious if not np.isnan(x)]
    return float(np.mean(valid)) if valid else float("nan")


# ─────────────────────────────────────────────
# Smoke test (--dry-run)
# ─────────────────────────────────────────────

def smoke_test(cfg, model: nn.Module, device: torch.device):
    """
    Verifica que el modelo funciona end-to-end con datos reales.

    Carga el primer plot de train, submuestrea 8192 puntos,
    hace forward + backward, imprime:
      - Input shape: (B, N, 3) + (B, N, 2)
      - Output shape: (B, N, 2)
      - Loss value
      - VRAM usada (torch.cuda.memory_allocated())
      - Tiempo de forward pass
    """
    print("Cargando plot de smoke test...")
    dataset_root = Path(cfg.paths.dataset_root)
    dev_paths, _ = load_splits(dataset_root)
    train_paths, _ = get_train_val_split(
        dev_paths,
        val_fraction=cfg.data.val_split,
        random_state=cfg.data.random_state,
    )

    # Dataset con 1 plot, sin augmentation para smoke test reproducible
    smoke_dataset = ForInstanceDataset(
        [train_paths[0]],
        num_points=cfg.pointnet2.num_points,
        augment=False,
        cfg=cfg,
    )

    points, labels, mask = smoke_dataset[0]  # (N,5), (N,), (N,)

    xyz = points[:, :3].unsqueeze(0).to(device)       # (1, N, 3)
    features = points[:, 3:].unsqueeze(0).to(device)  # (1, N, 2)
    labels_gpu = labels.unsqueeze(0).to(device)        # (1, N)

    print(f"Construyendo PointNet++ MSG...")
    n_params = count_parameters(model)
    print(f"  Parámetros totales: {n_params:,} (~{n_params/1e6:.1f}M)")

    # Pesos de clase para loss (estimados del smoke plot)
    all_labels = labels.numpy()
    counts = np.bincount(all_labels, minlength=2)
    total = counts.sum()
    weights = torch.FloatTensor(total / (2.0 * counts + 1e-6)).to(device)
    criterion = nn.NLLLoss(weight=weights)

    # Forward pass
    print("Forward pass...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    model.train()
    with autocast(
        device_type=device.type,
        dtype=_amp_dtype(device),
        enabled=cfg.pointnet2.mixed_precision,
    ):
        logits = model(xyz, features)  # (1, N, 2)
        loss = criterion(
            logits.reshape(-1, cfg.pointnet2.num_classes),
            labels_gpu.reshape(-1),
        )
    t_fwd = (time.time() - t0) * 1000

    print(f"  Input:  {tuple(xyz.shape)} + {tuple(features.shape)}")
    print(f"  Output: {tuple(logits.shape)}")
    print(f"  Loss:   {loss.item():.4f} (esperado ~{np.log(2):.3f} para red sin entrenar)")

    # Backward pass
    print("Backward pass...")
    loss.backward()
    all_grads_ok = all(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )
    print(f"  Gradientes: {'OK' if all_grads_ok else 'SOME MISSING'}")

    if device.type == "cuda":
        vram_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        vram_total_mb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
        print(f"VRAM usada: {vram_mb:.0f} MB / {vram_total_mb:.0f} MB disponibles")
        if vram_mb > 4000:
            print("  [WARN] VRAM > 4GB — considera reducir batch_size a 2 en config.yaml")
    elif device.type == "mps":
        # MPS no expone max_memory_allocated estable. Usa Activity Monitor o
        # `sudo powermetrics --samplers gpu_power` para medir externamente.
        print("Memoria: unified (MPS) — ver Activity Monitor para uso real")
    else:
        print("Memoria: N/A (CPU)")

    print(f"Tiempo forward: {t_fwd:.0f} ms")


# ─────────────────────────────────────────────
# Entrenamiento completo
# ─────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg,
    logger: logging.Logger,
) -> dict:
    """Entrena una época completa con AMP (bf16, sin GradScaler)."""
    model.train()
    total_loss = 0.0
    all_pred, all_true = [], []

    for points, labels, _mask in tqdm(loader, desc="  [train]", leave=False):
        xyz = points[:, :, :3].to(device)       # (B, N, 3)
        features = points[:, :, 3:].to(device)  # (B, N, 2)
        labels = labels.to(device)               # (B, N)

        optimizer.zero_grad()

        with autocast(
            device_type=device.type,
            dtype=_amp_dtype(device),
            enabled=cfg.pointnet2.mixed_precision,
        ):
            logits = model(xyz, features)        # (B, N, 2)
            loss = criterion(
                logits.reshape(-1, cfg.pointnet2.num_classes),
                labels.reshape(-1),
            )

        # bf16 mantiene el rango de exponente de fp32, así que no se necesita
        # loss scaling. GradScaler además es CUDA-only y rompería en MPS.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.pointnet2.grad_clip)
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        pred = logits.argmax(dim=-1).reshape(-1).cpu()
        all_pred.append(pred)
        all_true.append(labels.reshape(-1).cpu())

    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    miou = compute_miou(all_pred, all_true, cfg.pointnet2.num_classes)
    n_total = sum(len(p["labels"]) for p in loader.dataset.plots)

    return {
        "loss": total_loss / max(n_total, 1),
        "miou": miou,
        "acc": (all_pred == all_true).float().mean().item(),
    }


def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg,
) -> dict:
    """Evalúa en val set, retorna mIoU y loss."""
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []

    with torch.no_grad():
        for points, labels, _mask in tqdm(loader, desc="  [val]", leave=False):
            xyz = points[:, :, :3].to(device)
            features = points[:, :, 3:].to(device)
            labels = labels.to(device)

            with autocast(
                device_type=device.type,
                dtype=_amp_dtype(device),
                enabled=cfg.pointnet2.mixed_precision,
            ):
                logits = model(xyz, features)
                loss = criterion(
                    logits.reshape(-1, cfg.pointnet2.num_classes),
                    labels.reshape(-1),
                )

            total_loss += loss.item() * labels.numel()
            pred = logits.argmax(dim=-1).reshape(-1).cpu()
            all_pred.append(pred)
            all_true.append(labels.reshape(-1).cpu())

    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    miou = compute_miou(all_pred, all_true, cfg.pointnet2.num_classes)
    n_total = sum(len(p["labels"]) for p in loader.dataset.plots)

    return {
        "loss": total_loss / max(n_total, 1),
        "miou": miou,
        "acc": (all_pred == all_true).float().mean().item(),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento PointNet++ MSG")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Solo smoke test, no entrenamiento",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path a config.yaml (por defecto: forest_its/configs/config.yaml)",
    )
    args = parser.parse_args()

    # Cargar config
    if args.config:
        cfg = OmegaConf.load(args.config)
    else:
        cfg = OmegaConf.load(
            Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
        )

    # Reproducibilidad: fijar semillas antes de cualquier inicialización
    set_seed(int(cfg.data.random_state))

    device = select_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1024**2:.0f} MB")
    elif device.type == "mps":
        print("  Backend: Apple Silicon MPS (memoria unificada)")

    # Construir modelo
    model = PointNet2SemSegMSG(num_classes=cfg.pointnet2.num_classes).to(device)

    if args.dry_run:
        smoke_test(cfg, model, device)
        print()
        print("Smoke test pasado. Modelo listo para entrenamiento.")
        print("   Correr sin --dry-run manana para entrenamiento completo.")
        return

    # ────── Entrenamiento completo (para MAÑANA) ──────
    output_dir = Path(cfg.paths.output_dir)
    dataset_root = Path(cfg.paths.dataset_root)

    # Setup logging
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_pointnet2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_dir / "train_pointnet2.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    # Splits
    dev_paths, _ = load_splits(dataset_root)
    train_paths, val_paths = get_train_val_split(
        dev_paths,
        val_fraction=cfg.data.val_split,
        random_state=cfg.data.random_state,
    )
    logger.info(f"Train plots: {len(train_paths)}, Val plots: {len(val_paths)}")

    # Datasets
    logger.info("Loading train dataset...")
    train_dataset = ForInstanceDataset(train_paths, cfg.pointnet2.num_points, augment=True, cfg=cfg)
    logger.info("Loading val dataset...")
    val_dataset = ForInstanceDataset(val_paths, cfg.pointnet2.num_points, augment=False, cfg=cfg)

    nw_train = int(cfg.pointnet2.get("num_workers", 6))
    nw_val = int(cfg.pointnet2.get("num_workers_val", 4))

    # Generator para shuffle determinista del DataLoader
    seed = int(cfg.data.random_state)
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.pointnet2.batch_size,
        shuffle=True,
        num_workers=nw_train,
        persistent_workers=nw_train > 0,
        pin_memory=False,  # MPS no se beneficia de pinned memory
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.pointnet2.batch_size,
        shuffle=False,
        num_workers=nw_val,
        persistent_workers=nw_val > 0,
        pin_memory=False,
    )

    # Loss con pesos de clase
    class_weights = compute_class_weights(train_dataset).to(device)
    logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    criterion = nn.NLLLoss(weight=class_weights)

    # Optimizer y scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.pointnet2.lr,
        weight_decay=cfg.pointnet2.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.pointnet2.step_size, gamma=cfg.pointnet2.gamma,
    )
    # Paths de guardado
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / cfg.pointnet2.model_save_path.split("/")[-1]
    ckpt_path = models_dir / cfg.pointnet2.model_checkpoint_path.split("/")[-1]

    logger.info(f"Modelo: {count_parameters(model):,} parámetros")
    logger.info(f"Epochs: {cfg.pointnet2.epochs}, BS: {cfg.pointnet2.batch_size}, "
                f"LR: {cfg.pointnet2.lr}")

    best_val_miou = 0.0
    best_epoch = 0

    for epoch in range(1, cfg.pointnet2.epochs + 1):
        t_ep = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, cfg, logger)
        val_metrics = val_epoch(model, val_loader, criterion, device, cfg)

        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}/{cfg.pointnet2.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} mIoU: {train_metrics['miou']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} mIoU: {val_metrics['miou']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"t: {time.time() - t_ep:.0f}s"
        )

        # Checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_miou": val_metrics["miou"],
        }, ckpt_path)

        # Mejor modelo
        if val_metrics["miou"] > best_val_miou:
            best_val_miou = val_metrics["miou"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  -> Best model saved (val_mIoU={best_val_miou:.4f})")

    logger.info(f"\nTraining complete. Best epoch: {best_epoch} (val_mIoU={best_val_miou:.4f})")
    logger.info(f"Model saved: {best_model_path}")


if __name__ == "__main__":
    main()
