import numpy as np
import pandas as pd
from scipy.signal import butter
from scipy.signal import sosfilt
from scipy.stats import iqr, median_abs_deviation
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from extract_raw import pipe_data, extract_raw
import pickle

import config

class FeaturesExtract():

    def __init__(self):
        self.raw_df = pd.read_csv('./Data/RawExtract/raw.csv', index_col=0)
        # self.scalar = pickle.load(open(config.scalar_path, 'rb'))


    def calculate_features(self, signals):
        collate = []
        collate.append(np.mean(signals, axis=1))
        collate.append(np.std(signals, axis=1))
        collate.append(np.min(signals, axis=1))
        collate.append(np.max(signals, axis=1))
        collate.append(iqr(signals, axis=1))
        collate.append(median_abs_deviation(signals, axis=1))
        correlations = []
        for i in range(len(signals)):
            base = (i // 3) * 3
            correlations.append(np.correlate(signals[base + i % 3], signals[base + (i + 1) % 3])[0])
        collate.append(correlations)
        return np.concatenate(collate)

    def add_gravity_acc(self, signals):
        sos = butter(4, 0.3, output='sos', fs=config.fs)

        grav_acc = []
        for sig in signals[:3]:
            grav_acc.append(sosfilt(sos, sig))

        signals = np.concatenate((signals, grav_acc), axis=0)

        return signals


    def single_datapoint_get_features_no_transform(self, signals):
        signals = self.add_gravity_acc(signals)
        features = self.calculate_features(signals)
        # features = self.scalar.transform([features])
        return features

    def single_datapoint_get_features(self, signals):
        features = self.calculate_features(signals)
        # features = self.scalar.transform([features])
        return features

    def get_features_df(self):

        df = self.raw_df
        features_list = np.empty((df.shape[0], config.num_features))
        dancers_id = []
        moves_id = []

        for i in range(df.shape[0]):
            row = df.iloc[i, :-2]
            signals = row.to_numpy().flatten().reshape(config.num_axis, config.wl)
            # features = self.single_datapoint_get_features(signals)
            features = self.calculate_features(signals)
            features_list[i] = features
            dancers_id.append(df.iloc[i, -2])
            moves_id.append(df.iloc[i, -1])

        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_list)
        # normalized_features = self.scalar.transform(features_list)
        pickle.dump(scaler, open('./Data/RawExtract/scaler.pkl', 'wb'))
        self.save_scaler_npy(scaler)

        feature_df = pd.DataFrame(normalized_features)
        feature_df['dancer'] = dancers_id
        feature_df['label'] = moves_id

        feature_df.to_csv('./Data/RawExtract/features.csv')

    def save_scaler_npy(self, scaler):
        f_mean = './Data/RawExtract/scaler_mean.npy'
        with open(f_mean, 'wb') as f:
            np.save(f, scaler.mean_)
        f_std = './Data/RawExtract/scaler_std.npy'
        with open(f_std, 'wb') as f:
            np.save(f, scaler.scale_)

    @staticmethod
    def determine_position(curr_pos, movements):
        offset = 4  # number of movements
        movements = [m - offset for m in movements] # 0: left, 1: right: 2: standstill
        left = 0
        stand = 2

        stand_count = movements.count(stand)
        if stand_count == 1:
            stand_index = movements.index(stand)
            indices = [0, 1, 2]
            indices.remove(stand_index)
            curr_pos[indices[0]], curr_pos[indices[1]] = curr_pos[indices[1]], curr_pos[indices[0]]
        elif stand_count == 0:
            middle = movements[1]
            if middle == left:
                curr_pos.append(curr_pos.pop(0))
            else:
                curr_pos.insert(0, curr_pos.pop())
        print(curr_pos)
        return curr_pos





if __name__ == "__main__":
    pipe_data()
    # # extract_raw()
    fe = FeaturesExtract()
    fe.get_features_df()
    # FeaturesExtract.determine_position([2,3,1], [5,5,6])
    pass







