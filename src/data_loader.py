from datasets import load_dataset, load_from_disk
from src.paths import RAW_DATA_DIR, REPO_ID
from torch.utils.data import Dataset, TensorDataset, DataLoader as TorchDataLoader
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

class HuggingFaceImageDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item['image']

        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.uint8(img))

        x = self.transform(img)
        y = torch.tensor(item['label'], dtype=torch.long)
        return x, y

class DataLoader:
    def __init__(self, test_size = 0.2, batch_size = 32, seed = 42, num_workers = 2):
        self.local_path = RAW_DATA_DIR
        self.repository_id = REPO_ID
        self.test_size = test_size
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

    def _raw_is_empty(self) -> bool:
        return not any(f for f in self.local_path.iterdir() if not f.name.startswith("."))

    def download(self):
        # only download if there's not data in raw folder
        if self._raw_is_empty():
            dataset = load_dataset(self.repository_id)
            dataset.save_to_disk(str(self.local_path))

    def load(self):
        if self._raw_is_empty():
            raise FileNotFoundError("Data folder is empty")

        return load_from_disk(str(self.local_path))

    def get_train_data(self):
        train_data = self.load()['train']

        X_train = train_data['image']
        y_train = train_data['label']

        return X_train, y_train

    def get_train_loader(self):
        train_data = self.load()['train']
        dataset = HuggingFaceImageDataset(train_data)

        return TorchDataLoader(
                dataset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                pin_memory = True
                )

    def get_test_data(self):
        test_data = self.load()['test']

        X_test = test_data['image']
        y_test = test_data['label']

        return X_test, y_test

    def get_test_loader(self):
        test_data = self.load()['test']
        dataset = HuggingFaceImageDataset(test_data)

        return TorchDataLoader(
                dataset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers,
                pin_memory = True
                )
