"""
PointNet 3D Deep Learning Architecture (Qi et al., CVPR 2017).
Implements permutation-invariant point cloud feature extraction and classification:
- Spatial Transformer Networks (STN3d and STNkd)
- Shared MLPs via 1D Convolutions
- Symmetric Global Max Pooling
- Classification MLP Head with Dropout and Regularization
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d(nn.Module):
    """Spatial Transformer Network for 3D input coordinates (predicts 3x3 alignment matrix)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 3, N)
        batch_size = x.size(0)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        # Symmetric max pooling over points
        out = torch.max(out, 2, keepdim=True)[0]
        out = out.view(batch_size, 1024)

        out = F.relu(self.bn4(self.fc1(out)))
        out = F.relu(self.bn5(self.fc2(out)))
        out = self.fc3(out)

        # Initialize to identity matrix
        iden = torch.eye(3, dtype=x.dtype, device=x.device).view(1, 9).repeat(batch_size, 1)
        out = out + iden
        matrix = out.view(batch_size, 3, 3)
        return matrix


class STNkd(nn.Module):
    """Feature Transformer Network for intermediate feature space alignment (predicts k x k matrix)."""

    def __init__(self, k: int = 64):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, k, N)
        batch_size = x.size(0)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        out = torch.max(out, 2, keepdim=True)[0]
        out = out.view(batch_size, 1024)

        out = F.relu(self.bn4(self.fc1(out)))
        out = F.relu(self.bn5(self.fc2(out)))
        out = self.fc3(out)

        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        out = out + iden
        matrix = out.view(batch_size, self.k, self.k)
        return matrix


class PointNetEncoder(nn.Module):
    """PointNet feature encoder extracting global 1024-dim representation."""

    def __init__(self, global_feat: bool = True, feature_transform: bool = True):
        super().__init__()
        self.stn = STN3d()
        self.feature_transform = feature_transform
        if feature_transform:
            self.fstn = STNkd(k=64)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.global_feat = global_feat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Input shape: (B, 3, N)
        n_pts = x.size(2)

        # Coordinate transform
        trans = self.stn(x)
        x = torch.bmm(trans, x)

        out = F.relu(self.bn1(self.conv1(x)))

        # Feature transform
        if self.feature_transform:
            trans_feat = self.fstn(out)
            out = torch.bmm(trans_feat, out)
        else:
            trans_feat = None

        point_feat = out
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # Global max pool
        global_feat = torch.max(out, 2, keepdim=True)[0]
        global_feat = global_feat.view(-1, 1024)

        if self.global_feat:
            return global_feat, trans, trans_feat
        else:
            # Concatenate point and global features for segmentation
            global_feat_expanded = global_feat.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([point_feat, global_feat_expanded], 1), trans, trans_feat


class PointNetClassifier(nn.Module):
    """PointNet classification model predicting category logits."""

    def __init__(self, num_classes: int = 40, feature_transform: bool = True, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.feat = PointNetEncoder(global_feat=True, feature_transform=feature_transform)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # x shape: (B, 3, N) or (B, N, 3)
        if x.size(1) != 3 and x.size(2) == 3:
            x = x.transpose(2, 1)

        global_feat, trans, trans_feat = self.feat(x)

        out = F.relu(self.bn1(self.fc1(global_feat)))
        out = F.relu(self.bn2(self.dropout(self.fc2(out))))
        logits = self.fc3(out)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, trans, trans_feat


def feature_transform_regularizer(trans_feat: torch.Tensor) -> torch.Tensor:
    """Computes orthogonal regularization loss for the feature transformation matrix: ||I - AA^T||_F^2."""
    if trans_feat is None:
        return torch.tensor(0.0)
    d = trans_feat.size(1)
    iden = torch.eye(d, dtype=trans_feat.dtype, device=trans_feat.device).unsqueeze(0)
    diff = iden - torch.bmm(trans_feat, trans_feat.transpose(2, 1))
    return torch.mean(torch.norm(diff, dim=(1, 2)))

