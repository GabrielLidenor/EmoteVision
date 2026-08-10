import pytest
import torch
from src.models import FacialRecognitionModel

def test_model_output_shape():
    batch_size = 4
    num_classes = 5

    model = FacialRecognitionModel(num_classes = num_classes)

    fake_images = torch.randn(batch_size, 3, 150, 150)

    output = model(fake_images)

    assert isinstance(output, torch.Tensor), "Output must be a PyTorch Tensor"
    assert output.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {output.shape}"

def test_backbone_weights_are_frozen():
    model = FacialRecognitionModel()

    first_layer_param = next(model.base_model.parameters())

    assert first_layer_param.requires_grad is False, "Backbone parameters must be frozen"
    assert model.classifier.weight.requires_grad is True, "Classification head must be trainable"
