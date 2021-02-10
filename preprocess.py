import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import config as config


class HarDataset(Dataset):

    def __init__(self, txt_file, root_dir):
        self.txt_file = txt_file
        self.feature_frame = pd.read_table(txt_file, delim_whitespace=True, names=np.arange(561))

        self.root_dir = root_dir

    def __len__(self):
        return len(self.feature_frame)

    def __getitem__(self, idx):
        return self.feature_frame[idx]
