import os
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

try:
    from torchvision.models import ResNet18_Weights
except Exception:
    ResNet18_Weights = None


# =========================
# Adapter Module
# =========================
class Adapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 16):
        super().__init__()
        if bottleneck <= 0:
            raise ValueError(f"Adapter bottleneck must be > 0, got {bottleneck}")

        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)

    def forward(self, x):
        return x + self.up(F.relu(self.down(x)))


# =========================
# Main Model
# =========================
class SimpleCNN(nn.Module):
    """
    ResNet-18 backbone + adapters + task-local classifiers
    """

    def __init__(
        self,
        num_tasks: int = 10,
        num_classes: int = 100,
        bottleneck: int = 16,
        freeze_backbone_init: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()

        # -------------------------
        # Safety checks (NEW)
        # -------------------------
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be > 0, got {num_tasks}")

        if num_classes % num_tasks != 0:
            raise ValueError(
                f"num_classes must be divisible by num_tasks. "
                f"Got num_classes={num_classes}, num_tasks={num_tasks}"
            )

        if bottleneck <= 0:
            raise ValueError(f"bottleneck must be > 0, got {bottleneck}")

        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.classes_per_task = num_classes // num_tasks
        self.adapter_bottleneck = bottleneck

        # -------------------------
        # Backbone
        # -------------------------
        base = self._build_resnet18_backbone(pretrained)

        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        )

        self.feature_dim = 512

        # -------------------------
        # Adapters
        # -------------------------
        self.shared_adapter = Adapter(self.feature_dim, bottleneck)

        self.task_adapters = nn.ModuleList(
            [Adapter(self.feature_dim, bottleneck) for _ in range(num_tasks)]
        )

        # -------------------------
        # Task-local classifiers (CRITICAL FIX)
        # -------------------------
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.feature_dim, self.classes_per_task) for _ in range(num_tasks)]
        )

        # -------------------------
        # Optional backbone freeze
        # -------------------------
        if freeze_backbone_init:
            self.freeze_backbone()

        self._adapter_reference = None

    # =========================
    # Backbone Builder
    # =========================
    def _build_resnet18_backbone(self, pretrained: bool):
        if pretrained:
            try:
                if ResNet18_Weights is not None:
                    base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                else:
                    base = resnet18(pretrained=True)
            except Exception:
                base = resnet18(weights=None)
        else:
            base = resnet18(weights=None)

        # Modify for small images (CIFAR/Tiny-ImageNet)
        old_conv1 = base.conv1
        new_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        with torch.no_grad():
            if pretrained and old_conv1.weight.shape[-1] == 7:
                new_conv1.weight.copy_(old_conv1.weight[:, :, 2:5, 2:5])
            else:
                nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")

        base.conv1 = new_conv1
        base.maxpool = nn.Identity()

        return base

    # =========================
    # Backbone control
    # =========================
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    # =========================
    # Feature extraction
    # =========================
    def extract_features(self, x):
        feats = self.backbone(x)
        return torch.flatten(feats, 1)

    # =========================
    # Forward (SAFE VERSION)
    # =========================
    def forward(self, x, task_id=0, adapter_mode="shared"):
        task_id = int(task_id)

        if task_id < 0 or task_id >= self.num_tasks:
            raise IndexError(f"Invalid task_id {task_id}")

        feats = self.extract_features(x)

        if adapter_mode == "shared":
            feats = self.shared_adapter(feats)
        elif adapter_mode == "task":
            feats = self.task_adapters[task_id](feats)
        elif adapter_mode == "none":
            pass
        else:
            raise ValueError(f"Invalid adapter_mode: {adapter_mode}")

        return self.classifiers[task_id](feats)

    # =========================
    # Adapter utilities
    # =========================
    def get_adapter_params(self):
        return {k: v for k, v in self.named_parameters() if "adapter" in k}

    def snapshot_adapters(self):
        self._adapter_reference = {
            k: v.detach().clone() for k, v in self.get_adapter_params().items()
        }

    def get_adapter_delta(self):
        if self._adapter_reference is None:
            raise RuntimeError("Call snapshot_adapters() first")

        return {
            k: v.detach() - self._adapter_reference[k]
            for k, v in self.get_adapter_params().items()
        }

    def apply_adapter_delta(self, delta):
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name in delta:
                    p.add_(delta[name])