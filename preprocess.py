import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import config as config


class RawDataset(Dataset):

    def __init__(self):
        self.dataframe = pd.read_csv(config.raw_data_path, index_col=0)
        # self.get_first_300()
        self.signal_frame = self.dataframe.iloc[:, :(0-config.num_label_cols)]

    def __len__(self):
        return len(self.dataframe)

    def get_first_300(self):
        result = pd.DataFrame()

        for user in range(1, 30):
            user_dataframe = self.dataframe[self.dataframe['user_id'] == user]
            user_dataframe = user_dataframe.sample(n=300)
            result = pd.concat((result, user_dataframe))
        self.dataframe = result.reset_index(drop=True)


    def __getitem__(self, idx):
        features_img = self.signal_frame.iloc[idx, :].to_numpy()
        features_img = features_img.reshape(config.input_size, order='F').astype('float32')
        # features_img = np.expand_dims(features_img, axis=0)
        action = int(self.dataframe.loc[idx, 'label']) - 1
        sample = {'features': features_img, 'action': action}
        return sample



class FeatureDataset(Dataset):
    train_x_path = os.path.join('Data', 'Train', 'X_train.txt')
    train_y_path = os.path.join('Data', 'Train', 'Y_train.txt')
    test_x_path = os.path.join('Data', 'Test', 'X_test.txt')
    test_y_path = os.path.join('Data', 'Test', 'Y_test.txt')
    feature_names_path = os.path.join('Data', 'features.txt')

    def __init__(self, root_dir):

        self.features_names = pd.read_csv(self.feature_names_path, names=['feature_names']).to_numpy().flatten()
        self.train_x = pd.read_csv(self.train_x_path, delim_whitespace=True, names=np.arange(561))
        self.train_y = pd.read_csv(self.train_y_path, sep=" ", names=['label'])

        self.train_num = len(self.train_x)
        self.test_x = pd.read_csv(self.test_x_path, delim_whitespace=True, names=np.arange(561))
        self.test_y = pd.read_csv(self.test_y_path, sep=" ", names=['label'])

        self.features = pd.concat((self.train_x, self.test_x))
        self.features.columns = self.features_names
        self.labels = pd.concat((self.train_y, self.test_y))

        self.root_dir = root_dir

    

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features.iloc[idx, :].to_numpy().astype('float16')
        features = features.astype('float32')

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

    def get_shuffled_dataset(self):
        x_df = pd.concat((self.train_x, self.test_x))
        y_df = pd.concat((self.train_y, self.test_y))
        x_df.columns = self.features_names
        x_df = x_df.filter(regex='(?=.*(STD|Min|Max|Mean|IQR|Correlation|ARCoeff))^(tGravityAcc-|tBodyGyro-|tBodyAcc-)', axis=1)
        shuffle = pd.concat((x_df, y_df), axis=1).sample(frac=1)
        total_len = shuffle.shape[0]
        train_len = int(total_len * (1 - config.val_split))
        val_len = total_len - train_len

        train_x = shuffle.iloc[:train_len-1, :-1].to_numpy()
        train_y = shuffle.iloc[:train_len-1, -1:].to_numpy().flatten()
        test_x = shuffle.iloc[train_len:, :-1].to_numpy()
        test_y = shuffle.iloc[train_len:, -1:].to_numpy().flatten()

        return train_x, train_y, test_x, test_y


    def get_data_input(self):

        x_df = pd.concat((self.train_x, self.test_x))
        y_df = pd.concat((self.train_y, self.test_y))
        x_df.columns = self.features_names
        # x_df = x_df.filter(regex='(?=.*(STD|Min|Max|Mean|IQR|Correlation|ARCoeff))^(tGravityAcc-|tBodyGyro-|tBodyAcc-)',
        #                    axis=1)
        x_df = x_df.filter(regex='^t',
                           axis=1)
        shuffle = pd.concat((x_df, y_df), axis=1).sample(frac=1)
        X = shuffle.iloc[:, :-1].to_numpy()
        y = shuffle.iloc[:, -1:].to_numpy().flatten()
        return X, y



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




