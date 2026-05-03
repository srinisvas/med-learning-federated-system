
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def med_tiny_resnet18(num_classes: int = 8, base_width: int = 32) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    nn.init.kaiming_normal_(model.classifier[1].weight, mode="fan_out", nonlinearity="relu")
    nn.init.constant_(model.classifier[1].bias, 0.0)
    return model


if __name__ == "__main__":
    model = med_tiny_resnet18(num_classes=8)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape   :", y.shape)
    print("Total params   :", sum(p.numel() for p in model.parameters()))
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))