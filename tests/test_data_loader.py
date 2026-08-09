import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.data_loader import DataLoader
from datasets import Dataset, DatasetDict
from PIL import Image
import torch
from torch.utils.data import DataLoader as TorchDataLoader

@pytest.fixture
def mock_hugging_face_dataset():
    """Creates a dataset with 2 rows for test and train data"""

    train_image_1 = Image.new('RGB', (10, 10), color = 'red')
    train_image_2 = Image.new('RGB', (10,10), color = 'blue')

    test_image_1 = Image.new('RGB', (10, 10), color = 'red')
    test_image_2 = Image.new('RGB', (10,10), color = 'blue')

    train_dataset = Dataset.from_dict({
        'image': [np.array(train_image_1), np.array(train_image_2)],
        'label': [0,1]
        })

    test_dataset = Dataset.from_dict({
        'image': [np.array(test_image_1), np.array(test_image_2)],
        'label': [0,1]
        })

    return Dataset.from_dict({ 'train': train_dataset, 'test': test_dataset })

@patch('src.data_loader.load_from_disk')
def test_load_with_assets_in_folder(mock_load_from_disk, tmp_path):
    with patch('src.data_loader.RAW_DATA_DIR', tmp_path):
        loader = DataLoader()

        fake_file = tmp_path / "dummy.arrow"
        fake_file.parent.mkdir(parents = True, exist_ok = True)
        fake_file.touch()

        mock_dataset = MagicMock()
        mock_load_from_disk.return_value = mock_dataset

        result = loader.load()

    expected_string_path = str(tmp_path)

    mock_load_from_disk.assert_called_once_with(expected_string_path)
    assert result == mock_dataset

def test_load_with_no_data_in_folder(tmp_path):
    with patch('src.data_loader.RAW_DATA_DIR', tmp_path):
        data_loader = DataLoader()

        with pytest.raises(FileNotFoundError) as exec_info:
            data_loader.load()

@patch('src.data_loader.load_from_disk')
def test_get_train_data(mock_load_from_disk, tmp_path, mock_hugging_face_dataset):
    with patch('src.data_loader.RAW_DATA_DIR', tmp_path):
        loader = DataLoader()
        fake_file = tmp_path / "dummy.arrow"
        fake_file.parent.mkdir(parents = True, exist_ok = True)
        fake_file.touch()

        mock_load_from_disk.return_value = mock_hugging_face_dataset

        X_train, y_train = loader.get_train_data()

        assert len(X_train) == 2
        assert len(y_train) == 2

@patch('src.data_loader.load_from_disk')
def test_get_train_loader(mock_load_from_disk, tmp_path, mock_hugging_face_dataset):
    with patch('src.data_loader.RAW_DATA_DIR', tmp_path):
        expected_batch_size = 2
        expected_workers = 2

        data_loader = DataLoader(
                batch_size = expected_batch_size,
                num_workers = expected_workers
                )

        loader = DataLoader()
        fake_file = tmp_path / "dummy.arrow"
        fake_file.parent.mkdir(parents = True, exist_ok = True)
        fake_file.touch()

        mock_load_from_disk.return_value = mock_hugging_face_dataset

        train_loader = data_loader.get_train_loader()

        assert isinstance(train_loader, TorchDataLoader), "Returns a PyTorch DataLoader object"
        assert train_loader.batch_size == expected_batch_size, 'Batch size mismatch'
        assert train_loader.num_workers == expected_workers, 'Worker allocation mismatch'

        assert train_loader.drop_last is False

@patch('src.data_loader.load_from_disk')
def test_get_test_loader(mock_load_from_disk, tmp_path, mock_hugging_face_dataset):
    with patch('src.data_loader.RAW_DATA_DIR', tmp_path):
        expected_batch_size = 2
        expected_workers = 2

        data_loader = DataLoader(
                batch_size = expected_batch_size,
                num_workers = expected_workers
                )

        loader = DataLoader()
        fake_file = tmp_path / "dummy.arrow"
        fake_file.parent.mkdir(parents = True, exist_ok = True)
        fake_file.touch()

        mock_load_from_disk.return_value = mock_hugging_face_dataset

        test_loader = data_loader.get_test_loader()

        assert isinstance(test_loader, TorchDataLoader), "Returns a PyTorch DataLoader object"
        assert test_loader.batch_size == expected_batch_size, 'Batch size mismatch'
        assert test_loader.num_workers == expected_workers, 'Worker allocation mismatch'

        assert test_loader.drop_last is False
