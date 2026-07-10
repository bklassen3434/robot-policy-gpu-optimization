"""Image backbone.

This is the one component we deliberately reuse rather than build from scratch:
torchvision's ResNet-18, exactly as the official ACT does. The transformer is the
from-scratch signal; reimplementing a CNN adds noise, not proof.

We chop the classification head and return the final ``(B, 512, H/32, W/32)`` feature
map, which the ACT model projects to ``dim_model`` and flattens into tokens.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "torchvision is required for the image backbone. Install it via the "
                "project deps (pip install -e .)."
            ) from exc

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        # everything up to and including layer4, dropping avgpool + fc
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = 512

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (B, 512, H/32, W/32)


# ImageNet normalization constants (backbone was pretrained with these).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
