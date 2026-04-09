"""
PointNet++ MSG para segmentación semántica binaria.

Arquitectura base: Qi et al. (2017) 'PointNet++: Deep Hierarchical
Feature Learning on Point Sets in a Metric Space', NeurIPS 2017.
Implementación: yanx27/Pointnet_Pointnet2_pytorch (Yan, X., 2019),
https://github.com/yanx27/Pointnet_Pointnet2_pytorch
Modificaciones sobre el original: num_classes=2 (binario árbol/no-árbol),
in_channel=5 (XYZ + HAG + intensidad normalizada).
La adición de HAG e intensidad como features de entrada sigue a
Wang et al. (2023) Remote Sens., que demuestra mejora significativa
al añadir altura normalizada e intensidad sobre XYZ puro en
clasificación forestal con PointNet++.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from forest_its.methods.pointnet2.pointnet2_utils import (
    PointNetSetAbstractionMsg,
    PointNetFeaturePropagation,
)


class PointNet2SemSegMSG(nn.Module):
    """
    PointNet++ MSG para segmentación semántica binaria (árbol/no-árbol).

    Arquitectura base: Qi et al. (2017) 'PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space', NeurIPS 2017.
    Implementación: yanx27/Pointnet_Pointnet2_pytorch (Yan, X., 2019),
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    Modificaciones sobre el original: num_classes=2 (binario árbol/no-árbol),
    in_channel=5 (XYZ + HAG + intensidad normalizada).
    La adición de HAG e intensidad como features de entrada sigue a
    Wang et al. (2023) Remote Sens., que demuestra mejora significativa
    al añadir altura normalizada e intensidad sobre XYZ puro en
    clasificación forestal con PointNet++.

    Encoder (Set Abstraction con MSG):
      SA1: npoint=1024, radii=[0.1,0.2]m, nsamples=[16,32], in=5
      SA2: npoint=256,  radii=[0.2,0.4]m, nsamples=[16,32], in=96
      SA3: npoint=64,   radii=[0.4,0.8]m, nsamples=[16,32], in=256
      SA4: npoint=16,   radii=[0.8,1.6]m, nsamples=[16,32], in=512

    Decoder (Feature Propagation):
      FP4: in=1536, mlp=[512,512]
      FP3: in=768,  mlp=[512,256]
      FP2: in=352,  mlp=[256,128]
      FP1: in=133,  mlp=[128,128]  (skip con l0_points=5ch)

    Input:  xyz=(B,N,3) coordenadas + features=(B,N,2) [HAG_norm, intensity]
    Output: (B,N,num_classes) log-probabilidades por punto
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # === Encoder: Set Abstraction con MSG ===
        # Radios ajustados para UAV LiDAR forestal (metros).
        # in_channel=5: XYZ(3) + HAG_norm(1) + intensity(1).
        # Yanx27's PointNetSetAbstractionMsg: first conv input = in_channel + 3.
        # SA1: first conv = 5+3=8 → [16,16,32] y [32,32,64]. Output = 96ch.
        self.sa1 = PointNetSetAbstractionMsg(
            1024, [0.1, 0.2], [16, 32], 5,
            [[16, 16, 32], [32, 32, 64]]
        )
        # SA2: in=96 → first conv=99 → [64,64,128] y [64,96,128]. Output = 256ch.
        self.sa2 = PointNetSetAbstractionMsg(
            256, [0.2, 0.4], [16, 32], 32 + 64,
            [[64, 64, 128], [64, 96, 128]]
        )
        # SA3: in=256 → first conv=259 → [128,128,256] y [128,128,256]. Output = 512ch.
        self.sa3 = PointNetSetAbstractionMsg(
            64, [0.4, 0.8], [16, 32], 128 + 128,
            [[128, 128, 256], [128, 128, 256]]
        )
        # SA4: in=512 → first conv=515 → [256,256,512] y [256,384,512]. Output = 1024ch.
        self.sa4 = PointNetSetAbstractionMsg(
            16, [0.8, 1.6], [16, 32], 256 + 256,
            [[256, 256, 512], [256, 384, 512]]
        )

        # === Decoder: Feature Propagation ===
        # FP4: interp SA4(1024) → SA3(512). in=1024+512=1536, out=512.
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [512, 512])
        # FP3: interp FP4(512) → SA2(256). in=512+256=768, out=256.
        self.fp3 = PointNetFeaturePropagation(512 + 128 + 128, [512, 256])
        # FP2: interp FP3(256) → SA1(96). in=256+96=352, out=128.
        self.fp2 = PointNetFeaturePropagation(256 + 32 + 64, [256, 128])
        # FP1: interp FP2(128) → l0 skip(5). in=128+5=133, out=128.
        self.fp1 = PointNetFeaturePropagation(128 + 5, [128, 128])

        # === Clasificador final ===
        # Linear(128,64) → BN → ReLU → Dropout(0.5) → Linear(64,num_classes)
        self.conv1 = nn.Conv1d(128, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, num_classes, 1)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz:      (B, N, 3) coordenadas XYZ normalizadas.
            features: (B, N, 2) features adicionales [HAG_norm, intensity].

        Returns:
            (B, N, num_classes) log-probabilidades por punto.
        """
        # Convertir a formato canal-primero para yanx27's SA/FP layers: (B, C, N)
        xyz_cf = xyz.permute(0, 2, 1)        # (B, 3, N)
        feat_cf = features.permute(0, 2, 1)  # (B, 2, N)

        # l0: todos los canales de entrada como "points" (patrón yanx27)
        l0_xyz = xyz_cf                                          # (B, 3, N)
        l0_points = torch.cat([xyz_cf, feat_cf], dim=1)         # (B, 5, N)

        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)        # (B,3,1024) (B,96,1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)        # (B,3,256)  (B,256,256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        # (B,3,64)   (B,512,64)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)        # (B,3,16)   (B,1024,16)

        # Decoder (skip connections de las SA layers)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B,512,64)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B,256,256)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B,128,1024)
        # FP1 usa l0_points (5ch) como skip — l0_points aún no se sobrescribió
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)  # (B,128,N)

        # Clasificador
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))  # (B, 64, N)
        x = self.conv2(x)                                          # (B, num_classes, N)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)                                    # (B, N, num_classes)
        return x
