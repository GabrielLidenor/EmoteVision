import pytest
from src.models import FacialRecognitionModel
from src import Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class DummyTrainDataModule:
    """Minimal data model"""
    def get_train_loader(self):
        x = torch.randn(10,2)
        y = torch.randint(0, 2, (10,))

        return DataLoader(TensorDataset(x, y), batch_size = 2)

@pytest.fixture
def trainer_setup():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model = model, optimizer = optimizer, criterion = criterion)

    return trainer, model

def test_fit_with_no_data_provider(trainer_setup):
    trainer, _ = trainer_setup

    with pytest.raises(AttributeError):
        trainer.fit(data_provider = None)

def test_fit_with_a_valid_data_provider(trainer_setup):
    trainer, _ = trainer_setup
    dummy_data = DummyTrainDataModule()

    # completes the test without issues with one epoch
    trainer.fit(data_provider = dummy_data)

def test_fit_changes_model_weight(trainer_setup):
    trainer, model = trainer_setup
    dummy_data = DummyTrainDataModule()
    initial_weight = model.weight.clone()

    trainer.fit(data_provider = dummy_data, epochs = 1)

    assert not torch.equal(initial_weight, model.weight), "Model weights did not update during training!"
