import torch
import yaml
import json
from typing import Protocol, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DataProvider(Protocol):
    def get_test_loader(self) -> Any:
        ...

class Evaluator:
    def __init__(self, model: torch.nn.Module,data_provider: DataProvider):
        self.model = model
        self.data_provider = data_provider
        self._y_true = None
        self._y_pred = None

    @torch.no_grad()
    def _evaluate(self):
        if self._y_true is not None and self._y_pred is not None:
            return # prevents redundant interferance runs

        self.model.eval()
        loader = self.data_provider.get_test_loader()

        device = next(self.model.parameters()).device

        y_true_list = []
        y_pred_list = []

        for x, y in loader:
            x = x.to(device)
            outputs = self.model(x)
            predictions = torch.argmax(outputs, dim = 1)

            y_true_list.extend(y.cpu().numpy())
            y_pred_list.extend(predictions.cpu().numpy())

        self._y_true = np.array(y_true_list)
        self._y_pred = np.array(y_pred_list)

    def extract_y_true(self):
        loader = self.data_provider.get_test_loader()

        y_true = []

        for _, y in loader:
            y_true.extend(y.numpy() if hasattr(y, 'numpy') else y)

        return y_true

    def accuracy(self) -> float:
        self._evaluate()
        return float(accuracy_score(self._y_true, self._y_pred))

    def precision(self, average: str = 'macro') -> float:
        self._evaluate()
        return float(precision_score(self._y_true, self._y_pred, average = average))

    def recall(self, average: str = 'macro') -> float:
        self._evaluate()
        return float(recall_score(self._y_true, self._y_pred, average=average))

    def f1_score(self, average: str = 'macro') -> float:
        self._evaluate()
        return float(f1_score(self._y_true, self._y_pred, average=average))

    def plot_confusion_matrix(self, save_path: str = None):
        self._evaluate()
        cm = confusion_matrix(self._y_true, self._y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def get_classification_report(self) -> str:
        self._evaluate()
        return classification_report(self._y_true, self._y_pred)

    def export_metrics(self, filepath: str, format: str = 'json'):
        metrics = {
            "accuracy": self.accuracy(),
            "precision_macro": self.precision(average='macro'),
            "precision_weighted": self.precision(average='weighted'),
            "recall_macro": self.recall(average='macro'),
            "recall_weighted": self.recall(average='weighted'),
            "f1_macro": self.f1_score(average='macro'),
            "f1_weighted": self.f1_score(average='weighted'),
        }

        with open(filepath, 'w') as f:
            if format.lower() == 'json':
                json.dump(metrics, f, indent=4)
            elif format.lower() in ['yaml', 'yml']:
                yaml.dump(metrics, f)
            else:
                raise ValueError("Unsupported format. Choose 'json' or 'yaml'.")
