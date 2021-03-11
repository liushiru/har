import numpy as np
from scipy.signal import butter
from scipy.signal import sosfilt
from scipy.stats import iqr

import config

class FeaturesExtract():
    def __init__(self, signals):
        self.signals = self.add_gravity_acc(signals)
        self.features = self.calculate_features(self.signals)


    def get_all_features(self, signals):
        # signals = self.add_gravity_acc(signals)
        # features = self.calculate_features(self.signals)
        return self.features


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
            correlations.append(np.correlate(signals[base + i % 3], signals[base + (i + 1) % 3]))
        collate.append(correlations)

        return np.concatenate(collate)

    def add_gravity_acc(self, signals):
        sos = butter(4, 0.3, output='sos', fs=config.fs)

        grav_acc = []
        for sig in signals[:3]:
            grav_acc.append(sosfilt(sos, sig))

        signals = np.concatenate((signals, grav_acc), axis=0)

        return signals
