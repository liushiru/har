import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import config as config


class HarDataset(Dataset):
    train_x_path = os.path.join('Data', 'Train', 'X_train.txt')
    train_y_path = os.path.join('Data', 'Train', 'Y_train.txt')
    test_x_path = os.path.join('Data', 'Test', 'X_test.txt')
    test_y_path = os.path.join('Data', 'Test', 'Y_test.txt')

    def __init__(self, root_dir):
        self.train_x = pd.read_table(self.train_x_path, delim_whitespace=True, names=np.arange(561))
        self.train_y = pd.read_csv(self.train_y_path, sep=" ", names=['label'])

        self.train_num = len(self.train_x)
        self.test_x = pd.read_table(self.test_x_path, delim_whitespace=True, names=np.arange(561))
        self.test_y = pd.read_csv(self.test_y_path, sep=" ", names=['label'])


        self.features = pd.concat((self.train_x, self.test_x))
        self.labels = pd.concat((self.train_y, self.test_y))

        self.root_dir = root_dir

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features.iloc[idx, :].to_numpy().astype('float32')
        action = self.labels.iloc[idx].item()-1
        sample = {'features': features, 'action': action}
        return sample

    def normalize(self):
        df = self.features
        scaled_features = StandardScaler().fit_transform(df.values)
        self.features = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)


    def get_train_x(self):
        return self.train_x.to_numpy()

    def get_train_y(self):
        return self.train_y.to_numpy().flatten()

    def get_test_x(self):
        return self.test_x.to_numpy()

    def get_test_y(self):
        return self.test_y.to_numpy().flatten()