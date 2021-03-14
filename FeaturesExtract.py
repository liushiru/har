import numpy as np
import pandas as pd
from scipy.signal import butter
from scipy.signal import sosfilt
from scipy.stats import iqr
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle

import config

class FeaturesExtract():

    def __init__(self):
        self.raw_df = pd.read_csv('./Data/RawExtract/raw.csv', index_col=0)
        self.scalar = pickle.load(open(config.scalar_path, 'rb'))

    def detect_movement_start(self, packet):
        pass


    def calculate_features(self, signals):
        collate = []
        collate.append(np.mean(signals, axis=1))
        collate.append(np.std(signals, axis=1))
        collate.append(np.min(signals, axis=1))
        collate.append(np.max(signals, axis=1))
        collate.append(iqr(signals, axis=1))
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


    def single_datapoint_get_features(self, signals):
        signals = self.add_gravity_acc(signals)
        features = self.calculate_features(signals)
        features = self.scalar.transform([features])
        return features[0]

    def get_features_df(self):

        df = self.raw_df
        col_name = [i for i in range(54)]
        features_list = np.empty((df.shape[0], 54))
        dancers_id = []
        moves_id = []

        for i in range(df.shape[0]):
            row = df.iloc[i, :-2]
            signals = row.to_numpy().flatten().reshape(6, config.wl)
            features = self.single_datapoint_get_features(signals)
            features_list[i] = features
            dancers_id.append(df.iloc[i, -2])
            moves_id.append(df.iloc[i, -1])



        scalar = StandardScaler()
        normalized_features = scalar.fit_transform(features_list)
        pickle.dump(scalar, open('./Data/RawExtract/scalar.pkl', 'wb'))
        feature_df = pd.DataFrame(normalized_features)
        feature_df['dancer'] = dancers_id
        feature_df['label'] = moves_id



        feature_df.to_csv('./Data/RawExtract/features.csv')



if __name__ == "__main__":
    fe = FeaturesExtract()
    fe.get_features_df()

