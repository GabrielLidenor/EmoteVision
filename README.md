[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![Hugging Face Datasets](https://img.shields.io/badge/Hugging%20Face-Datasets-blue.svg)](https://huggingface.co/)
[![GPU Support](https://img.shields.io/badge/Acceleration-CUDA%20%7C%20MPS-yellowgreen.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-PyTest-green.svg)](tests/)


# EmoteVision

A compact end-to-end training and evaluation pipeline for facial expression classification using a Hugging Face dataset and a ResNet50 backbone.

## Installation

1. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

## Quick run

Run the full pipeline (download dataset, train, evaluate, export metrics and plots):

```bash
python main.py --epochs 1 --batch_size 8
```

Configuration options are available via CLI flags in `main.py` or by calling `run_pipeline(config)` in `src/pipeline.py`.

Downloaded dataset artifacts are saved to `data/raw` by `src/data_loader.DataLoader.download()` using Hugging Face `datasets.save_to_disk()`.

## Running tests

Unit tests use `pytest` and are located in the `tests/` folder. Run them with:

```bash
pytest -q
```

Key tests:
- `tests/test_data_loader.py`: dataset download/load and DataLoader constructions (uses mocks)
- `tests/test_facial_recognition.py`: model shapes and frozen backbone checks
- `tests/test_trainer.py`: training loop behaviours

## Project layout

Top-level files and folders:

- `main.py`: CLI entrypoint to run the pipeline
- `src/`: application code
    - `data_loader.py`: dataset download/persistence and PyTorch `DataLoader` wrapping
    - `paths.py`: project path constants (e.g., `data/raw`)
    - `pipeline.py`: orchestrates data download, training, evaluation, artifact export
    - `trainer.py`: training loop that consumes a `DataProvider` (returns a PyTorch `DataLoader`)
    - `evaluator.py`: evaluation helpers and artifact export (metrics, confusion matrix)
    - `models/`: model definitions (ResNet50 backbone + classifier head)
- `data/`: storage for raw and processed datasets
   - `data/raw/`: persisted Hugging Face dataset (created by `save_to_disk()`)
   - `data/processed/`: optional processed artifacts
- `outputs/`: saved artifacts: `metrics.json`, `confusion_matrix.png`, etc.
- `tests/`: unit tests
- `requirements.txt` and `requirements-dev.txt`

## Architecture & data flow

1. Data download & persistence
   - `DataLoader.download()` calls `datasets.load_dataset(REPO_ID)` and then `dataset.save_to_disk(data/raw)`.
2. Data loading & transforms
   - `DataLoader.load()` uses `load_from_disk(data/raw)`.
   - `HuggingFaceImageDataset` converts HF rows to PIL/Numpy images, applies `Grayscale -> ToTensor -> Normalize` transforms, and returns `(image_tensor, label)`.
3. Model
   - Backbone: pretrained `resnet50` (most layers frozen except `layer4` by default).
   - Embedding head: `Linear(in_features, embedding_size)` followed by `ReLU`.
   - Classifier head: `Linear(embedding_size, num_classes)` returning logits for `CrossEntropyLoss`.
4. Training
   - `Trainer.fit()` fetches `train_loader` and runs forward → loss (`CrossEntropyLoss`) → backward → optimizer.step().
5. Evaluation
   - `Evaluator` runs model on the test loader, computes metrics and writes `outputs/metrics.json` and `outputs/confusion_matrix.png`.

## Rationale: classification head & choice of loss

This project treats facial expression recognition as a supervised multi-class classification task because the dataset provides per-image categorical labels (e.g., happy, sad, angry). The model uses a small classifier head on top of a pretrained ResNet50 backbone and is trained with `nn.CrossEntropyLoss`. Reasons for this design:

- **Direct supervision and metrics:** `CrossEntropyLoss` expects raw class logits and pairs naturally with evaluation metrics like accuracy, precision, and F1, making progress easy to interpret.
- **Numerical stability and simplicity:** `CrossEntropyLoss` implements `log_softmax` + `nll_loss` in a stable, optimized form and is the standard choice for multi-class classification.
- **Practicality and reproducibility:** A classification head requires less engineering than metric-learning pipelines (which need careful positive/negative mining or contrastive sampling) and trains efficiently using standard PyTorch optimizers.

Alternative (embedding / metric-learning) approaches have advantages for retrieval, few-shot learning, or when labels are unreliable, but they require different losses (contrastive, triplet, NT-Xent), different sampling strategies, and different evaluation protocols. I opted for a classifier-first approach to match the dataset's supervised labels and to keep the pipeline simple and reproducible.

Common causes for unexpectedly large loss (what to check):

- **Model-output vs. loss mismatch:** returning normalized embeddings while using `CrossEntropyLoss` will produce meaningless loss values. Ensure model outputs are logits with shape `(batch_size, num_classes)`.
- **Label issues:** verify labels are integer `torch.long` values in the range `[0, num_classes-1]`.
- **Shape/dtype mismatches:** confirm `outputs.shape` and `targets.shape` match expectations and dtypes are correct.
- **Data problems:** corrupted images, missing data, or incorrect normalization can destabilize training.
- **Optimization settings:** too-large learning rates, incorrect optimizer setup, or missing gradient zeroing can cause loss explosion.
- **Numerical instability:** NaNs in inputs/outputs or extremely large activations (inspect `torch.isnan()` and output statistics).

If you'd like to experiment with embeddings instead, I can add a configurable option to switch between `CrossEntropyLoss` and a metric loss, implement simple contrastive sampling, or add runtime checks/logging to `Trainer.fit()` to surface the most common issues.


## Debugging tips

- Print shapes and types for a single batch:

```py
print(inputs.shape, inputs.dtype)
print(targets.shape, targets.dtype, targets.min(), targets.max())
```

- Inspect model outputs:

```py
o = model(inputs)
print(o.shape, o.mean().item(), o.std().item(), torch.isnan(o).any())
```

- Check loss value for a single batch:

```py
loss = criterion(o, targets)
print(loss.item())
```