import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from src.emote_vision_quantum_head import EmoteVisionQuantumHead


class EmoteVisionQuantumModel(nn.Module):
    def __init__(self, number_of_classes: int = 4, run_on_ibm: bool = False):
        super().__init__()

        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model_features = self.base_model.fc.in_features

        for name, param in self.base_model.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False

        self.base_model.fc = EmoteVisionQuantumHead(
            model_features,
            number_of_layers = 10,
            number_of_qubits = 11,
            number_of_classes = number_of_classes,
            run_on_ibm = run_on_ibm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits suitable for nn.CrossEntropyLoss."""
        return self.base_model(x)
 