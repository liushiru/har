import pickle

import torch
import numpy as np
from torch.utils.data import DataLoader

import config
from cnn_main import get_dataloaders
from model import CNN
from preprocess import RawDataset, FeatureDataset


def cnn_inference():
    model = CNN()
    model.load_state_dict(torch.load(config.cnn_model_path))
    model.eval()

    dataset = RawDataset()
    dataloader = DataLoader(dataset, batch_size=1,
                                     shuffle=True, num_workers=4)

    with torch.no_grad():
        for data in dataloader:
            inputs = data['features']
            labels = data['action']
            outputs = model(inputs)
            probability_distribution = torch.nn.functional.softmax(outputs)
            prediction = np.argmax(probability_distribution.detach().numpy())
            print('prediction of CNN model is {}'.format(prediction))
            print('label is {}'.format(labels.detach().numpy()[0]))
            print('----')


def svm_inference():
    dataset = FeatureDataset()
    X, y = dataset.get_data_input()

    model = pickle.load(open(config.svm_model_path, "rb"))
    for idx in range(len(y)):
        prediction = model.predict([X[idx]])
        label = y[idx]
        print('prediction of SVM is {}'.format(prediction))
        print('label is {}'.format(label))
        print('--------')


if __name__ == "__main__":

    # cnn_inference()
    svm_inference()
