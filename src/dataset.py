import polars as pl
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Callable


class AnnotatedDataset(Dataset):
    def __init__(
        self,
        annotations_file: str | Path,
        test_columns: str | list[str],
        transform: Callable | None = None,
    ) -> None:
        annotations_file = Path(annotations_file)
        if isinstance(test_columns, str):
            test_columns = [test_columns]

        self._root = annotations_file.parent
        self.annotations = pl.read_csv(annotations_file)
        self.test_columns = test_columns
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        row = self.annotations.row(idx, named=True)
        result = {}
        for k in self.test_columns:
            v = row[k]
            if k.endswith('Path'):
                img = Image.open(self._root / v).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                result[k.replace('Path', 'Image')] = img
                result[k] = str(self._root / v)
            else:
                result[k] = v
        return result
