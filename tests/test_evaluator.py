import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src import Evaluator

class DummyTestDataModule:
    """Minimal data model"""
    def get_test_loader(self):
        x = torch.randn(10,2)
        y = torch.randint(0, 2, (10,))

        return DataLoader(TensorDataset(x, y), batch_size = 2)

@pytest.fixture
def evaluator_setup():
    model = torch.nn.Linear(2, 2)
    evaluator = Evaluator(model = model, data_provider = DummyTestDataModule())

    return evaluator, model


def test_accuracy_with_invalid_test_dataset(evaluator_setup):
    _, model = evaluator_setup
    evaluator = Evaluator(model = model, data_provider = None)

    with pytest.raises(AttributeError):
        evaluator.accuracy()

def test_accuracy_with_valid_data(evaluator_setup):
    _, model = evaluator_setup
    evaluator = Evaluator(model = model, data_provider = DummyTestDataModule())

    assert isinstance(evaluator.accuracy(), float)

def test_precision_with_invalid_test_dataset(evaluator_setup):
    _, model = evaluator_setup
    evaluator = Evaluator(model = model, data_provider = None)

    with pytest.raises(AttributeError):
        evaluator.precision()

def test_precision_with_valid_data(evaluator_setup):
    _, model = evaluator_setup
    evaluator = Evaluator(model = model, data_provider = DummyTestDataModule())

    assert isinstance(evaluator.precision(), float)

def test_recall_with_invalid_test_dataset(evaluator_setup):
    _, model = evaluator_setup
    evaluator = Evaluator(model = model, data_provider = None)

    with pytest.raises(AttributeError):
        evaluator.recall()

def test_recall_with_valid_data(evaluator_setup):
    _, model = evaluator_setup
    evaluator = Evaluator(model = model, data_provider = DummyTestDataModule())

    assert isinstance(evaluator.recall(), float)

def test_precision_f1_with_invalid_test_dataset(evaluator_setup):
    _, model = evaluator_setup
    evaluator = Evaluator(model = model, data_provider = None)

    with pytest.raises(AttributeError):
        evaluator.f1_score()

def test_precision_f1_with_valid_data(evaluator_setup):
    _, model = evaluator_setup
    evaluator = Evaluator(model = model, data_provider = DummyTestDataModule())

    assert isinstance(evaluator.f1_score(), float)

def test_get_classification_report(evaluator_setup):
    _, model = evaluator_setup
    evaluator = Evaluator(model = model, data_provider = DummyTestDataModule())
    report = evaluator.get_classification_report()

    assert isinstance(report, str)
    assert "precision" in report
    assert "recall" in report
    assert "f1-score" in report
    assert "accuracy" in report

def test_export_metrics_json(tmp_path, evaluator_setup):
    _, model = evaluator_setup
    evaluator = Evaluator(model = model, data_provider = DummyTestDataModule())
    json_file = tmp_path / "metrics.json"

    evaluator.export_metrics(str(json_file), format="json")

    assert json_file.exists()

def test_export_metrics_yaml(tmp_path, evaluator_setup):
    _, model = evaluator_setup
    evaluator = Evaluator(model = model, data_provider = DummyTestDataModule())
    yaml_file = tmp_path / "metrics.yaml"

    evaluator.export_metrics(str(yaml_file), format="yaml")

    assert yaml_file.exists()

def test_export_metrics_with_invalid_format(evaluator_setup, tmp_path):
    _, model = evaluator_setup
    evaluator = Evaluator(model = model, data_provider = DummyTestDataModule())

    with pytest.raises(ValueError):
        evaluator.export_metrics(filepath = tmp_path / "metrics.json", format = 'cobol')
