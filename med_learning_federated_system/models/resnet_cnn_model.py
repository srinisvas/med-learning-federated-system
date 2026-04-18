import torch
import torch.nn as nn
from typing import Optional


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def _make_layer(in_planes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
    downsample = None
    if stride != 1 or in_planes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )
    layers = [BasicBlock(in_planes, planes, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))
    return nn.Sequential(*layers)


class MedTinyResNet18(nn.Module):
    """
    ResNet-18 layout (2-2-2-2 blocks) with ImageNet-style 7x7 stem,
    suitable for 224x224 inputs and configurable via base_width.

    base_width=32 -> ~4.2M params   (recommended for ISIC)
    base_width=16 -> ~1.05M params  (constrained compute)
    base_width=8  -> ~0.27M params  (very constrained, likely underfits ISIC)
    """

    def __init__(self, num_classes: int = 8, base_width: int = 32):
        super().__init__()
        bw = base_width
        self.in_planes = bw

        # ImageNet-style stem: 224 -> 112 -> 56
        self.stem = nn.Sequential(
            nn.Conv2d(3, bw, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(bw),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 112 -> 56
        )

        # Four stages — spatial resolution: 56 -> 56 -> 28 -> 14 -> 7
        self.layer1 = _make_layer(bw,    bw,    blocks=2, stride=1)
        self.layer2 = _make_layer(bw,    bw*2,  blocks=2, stride=2)
        self.layer3 = _make_layer(bw*2,  bw*4,  blocks=2, stride=2)
        self.layer4 = _make_layer(bw*4,  bw*8,  blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(bw * 8, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.momentum = 0.05

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def med_tiny_resnet18(num_classes: int = 8, base_width: int = 32) -> MedTinyResNet18:
    return MedTinyResNet18(num_classes=num_classes, base_width=base_width)


if __name__ == "__main__":
    model = med_tiny_resnet18()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)          # [2, 8]
    print("Params:", sum(p.numel() for p in model.parameters()))  # ~4.2M at bw=32
