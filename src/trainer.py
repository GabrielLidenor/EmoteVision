from typing import Protocol, Any
import torch

class DataProvider(Protocol):
    def get_train_loader(self) -> Any:
        ...

class Trainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: Any):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def fit(self, data_provider: DataProvider, epochs: int = 5):
        train_loader = data_provider.get_train_loader()

        for epoch in range(epochs):
            self.model.train()

            for batch in train_loader:
                # the inputs are the images, and the targets are the labels (the answers )
                inputs, targets = batch
                # clears the stored gradients from the previous step
                # necessary because PyTorch accumulates the weights on backward
                # if we don't clean this up, it will explode the weights corrupting the training
                self.optimizer.zero_grad()
                # feeds the model with inputs through model's layers to generate new model predictions
                outputs = self.model(inputs)
                # compares models outputs with the answers
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

