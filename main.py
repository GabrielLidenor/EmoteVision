import argparse
from src.pipeline import run_pipeline
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run End-to-End ML Pipeline")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--use_quantum", action="store_true")
    parser.add_argument("--run_on_ibm", action="store_true")

    args = parser.parse_args()

    config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "output_dir": "outputs",
        "use_quantum": args.use_quantum,
        "run_on_ibm": args.run_on_ibm,
    }

    run_pipeline(config)
