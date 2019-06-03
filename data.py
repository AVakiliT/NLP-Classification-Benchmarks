import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

describe

class TextDataset(Dataset):

    def __getitem__(self, index: int) -> T_co:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()

    def __init__(self) -> None:
        super().__init__()