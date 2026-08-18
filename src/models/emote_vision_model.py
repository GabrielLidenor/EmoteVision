import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class EmoteVisionModel(nn.Module):
    def __init__(self, embedding_size: int = 512, num_classes: int = 7):
        super().__init__()

        # Pretrained ResNet50 backbone
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        for name, param in self.base_model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.embedding_layer = nn.Linear(in_features, embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the execution pipeline for processing data batches.
        Returns logits suitable for `nn.CrossEntropyLoss`.
        """
        features = self.base_model(x)
        vector = torch.flatten(features, start_dim=1)

        embeddings = self.embedding_layer(vector)
        embeddings = F.relu(embeddings)

        logits = self.classifier(embeddings)

        return logits
