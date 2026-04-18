import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def med_tiny_resnet18(num_classes: int = 8, base_width: int = 32) -> nn.Module:

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features      # 512 for ResNet18
    model.fc = nn.Linear(in_features, num_classes)
    # Kaiming init on the new head — rest of the model keeps pretrained weights
    nn.init.kaiming_normal_(model.fc.weight, mode="fan_out", nonlinearity="relu")
    nn.init.constant_(model.fc.bias, 0.0)
    return model


if __name__ == "__main__":
    model = med_tiny_resnet18(num_classes=8)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
    print("Total params:", sum(p.numel() for p in model.parameters()))
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))