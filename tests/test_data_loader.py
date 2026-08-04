import pytest
from unittest.mock import MagicMock, patch
from src.data_loader import DataLoader

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

def test_load_with_no_data_in_folder():
    loader = DataLoader()

    with pytest.raises(FileNotFoundError) as exec_info:
        loader.load()
