import copy
import os
import time
import argparse

from preprocess import HarDataset

if __name__ == "__main__":
    dataset = HarDataset(txt_file=os.path.join('Data', 'Train', 'X_train.txt'),
                         root_dir="Data")
    print(dataset)