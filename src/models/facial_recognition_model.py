import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class FacialRecognitionModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.num_classes = num_classes

        # Pretrained ResNet50 backbone
        self.base_model = resnet50(weights = ResNet50_Weights.DEFAULT)

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        # ResNet50 is a pretrained model, which reduces the amount
        # of time we need to train with our data.
        # we freeze the layers to prevent our data to make our model
        # to forget what they pre-trained before.
        # we should only focus on data that the model cannot read with
        # its pre-trained data
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.dense1 = nn.Linear(in_features, 256)
        self.dense2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, self.num_classes)

        self.relu = nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the execution pipeline for processing data batches.
        """
        features = self.base_model(x)
        vector = torch.flatten(features, start_dim = 1)

        embedding1 = self.relu(self.dense1(vector))
        embedding2 = self.relu(self.dense2(embedding1))

        predictions = self.classifier(embedding2)

        return predictions

