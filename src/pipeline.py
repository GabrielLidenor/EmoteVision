import os
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.panel import Panel

from src import DataLoader, Trainer, Evaluator
from src.models import FacialRecognitionModel

console = Console()

def run_pipeline(config: dict = None) -> None:
    if config is None:
        config = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10,
            "output_dir": "outputs"
        }

    output_dir = config.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    console.print(Panel.fit("[bold blue]Facial Expression Recognition Pipeline[/bold blue]"))

    # Step 1: Data Preparation
    with console.status("[bold green]Loading DataModule...[/bold green]", spinner="dots"):
        data_module = DataLoader(batch_size=config["batch_size"])
        data_module.download()
    console.print("  Loaded Data Module")

    # Step 2: Model & Optimizer Setup
    with console.status("[bold green]Initializing Model & Optimizer...[/bold green]", spinner="dots"):
        model = FacialRecognitionModel()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = nn.CrossEntropyLoss()
    console.print("  Initialized Model and Optimizer")

    # Step 3: Model Training
    console.print("\n[bold yellow]Starting Training Loop...[/bold yellow]")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config["epochs"]
    )
    trainer.fit(data_provider=data_module)
    console.print("  Training Complete")

    # Step 4: Model Evaluation
    console.print("\n[bold yellow]Running Evaluation & Generating Artifacts...[/bold yellow]")
    evaluator = Evaluator(model=model, data_provider=data_module)

    metrics_path = os.path.join(output_dir, "metrics.json")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")

    evaluator.export_metrics(metrics_path, format="json")
    evaluator.plot_confusion_matrix(save_path=cm_path)

    console.print(f"  Metrics exported to: [underline]{metrics_path}[/underline]")
    console.print(f"  Confusion Matrix saved to: [underline]{cm_path}[/underline]")

    console.print("\n[bold green]Pipeline execution finished successfully![/bold green]")
