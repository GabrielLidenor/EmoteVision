from typing import Protocol, Any
import torch
from tqdm import tqdm

class DataProvider(Protocol):
    def get_train_loader(self) -> Any:
        ...

class Trainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: Any, epochs: int = 5, scheduler: Any = None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.scheduler = scheduler

    def fit(self, data_provider: DataProvider):
        train_loader = data_provider.get_train_loader()

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

            for batch in pbar:
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

                running_loss += loss.item()

                pbar.set_postfix(loss=f"{loss.item():.4f}")

                avg_loss = running_loss / len(train_loader)

                if self.scheduler is not None:
                    self.scheduler.step(avg_loss)

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}] - Loss: {avg_loss:.4f}")
