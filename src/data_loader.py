from datasets import load_dataset, load_from_disk
from src.paths import RAW_DATA_DIR, REPO_ID

class DataLoader:
    def __init__(self):
        self.local_path = RAW_DATA_DIR
        self.repository_id = REPO_ID

    def download(self):
        dataset = load_dataset(self.repository_id)

        dataset.save_to_disk(str(self.local_path))

    def load(self):
        if not any(self.local_path.iterdir()):
            raise FileNotFoundError("Data folder is empty")

        return load_from_disk(str(self.local_path))

