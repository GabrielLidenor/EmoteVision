from datasets import load_dataset, load_from_disk
from src.paths import RAW_DATA_DIR, REPO_ID
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

class DataLoader:
    def __init__(self, test_size = 0.2, batch_size = 32, seed = 42, num_workers = 2):
        self.local_path = RAW_DATA_DIR
        self.repository_id = REPO_ID
        self.test_size = test_size
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

    def download(self):
        dataset = load_dataset(self.repository_id)

        dataset.save_to_disk(str(self.local_path))

    def load(self):
        if not any(self.local_path.iterdir()):
            raise FileNotFoundError("Data folder is empty")

        return load_from_disk(str(self.local_path))

    def get_train_data(self):
        train_data = self.load()['train']

        X_train = train_data['image']
        y_train = train_data['label']

        return X_train, y_train

    def get_train_loader(self):
        X_train, y_train = self.get_train_data()

        X_tensor = torch.tensor(X_train, dtype = torch.float32)
        y_tensor = torch.tensor(y_train, dtype = torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)

        return TorchDataLoader(
                dataset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers
                )

    def get_test_data(self):
        test_data = self.load()['test']

        X_test = test_data['image']
        y_test = test_data['label']

        return X_test, y_test

    def get_test_loader(self):
        X_test, y_test = self.get_test_data()

        X_tensor = torch.tensor(X_test, dtype = torch.float32)
        y_tensor = torch.tensor(y_test, dtype = torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)

        return TorchDataLoader(
                dataset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers
                )
