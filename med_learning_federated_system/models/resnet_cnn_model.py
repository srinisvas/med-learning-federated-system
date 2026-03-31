"""TinyResNet18 with configurable width — drop-in from the CIFAR experiment."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False,
        )
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False,
        )
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class TinyResNet18(nn.Module):
    """
    ResNet-18 with a width multiplier (`base_width`).

    base_width=8  gives  [8, 16, 32, 64] channels — the same as the
    CIFAR experiment, and compact enough for CPU-only simulation.
    """

    def __init__(self, num_classes: int = 10, base_width: int = 8):
        super().__init__()
        w = base_width
        self.in_planes = w

        self.conv1 = nn.Conv2d(3, w, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(w)

        self.layer1 = self._make_layer(w,      2, stride=1)
        self.layer2 = self._make_layer(w * 2,  2, stride=2)
        self.layer3 = self._make_layer(w * 4,  2, stride=2)
        self.layer4 = self._make_layer(w * 8,  2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(w * 8 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def tiny_resnet18(num_classes: int = 10, base_width: int = 8) -> TinyResNet18:
    """Factory function matching the original project's import signature."""
    return TinyResNet18(num_classes=num_classes, base_width=base_width)