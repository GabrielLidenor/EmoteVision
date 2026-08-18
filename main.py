import argparse
from src.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run End-to-End ML Pipeline")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)

    args = parser.parse_args()

    config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "output_dir": "outputs"
    }

    run_pipeline(config)
