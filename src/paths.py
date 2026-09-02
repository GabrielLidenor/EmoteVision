from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# data folder
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# artifacts folder
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# hugging face project data
REPO_ID = "gabriellidenor/facial_emotion_images"

