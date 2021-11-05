import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2


class FacesDataset(Dataset):
    def __init__(self, folder: str, transforms: A.Compose):
        self.folder = Path(folder)
        self.table = pd.DataFrame(
            pd.Series([file.name for file in self.folder.glob('*.png')], name='fname')
        )
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = cv2.imread(str(self.folder / self.table.loc[idx, 'fname']))[:, :, ::-1]
        return self.transforms(image=img)['image']
