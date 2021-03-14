import pickle

import torch
import numpy as np
from torch.utils.data import DataLoader

import config
from cnn_main import get_dataloaders
from model import CNN, MLP
from preprocess import RawDataset, FeatureDataset


def cnn_inference():
    model = MLP()
    model.load_state_dict(torch.load(config.cnn_model_path))
    model.eval()

    dataset = FeatureDataset()
    dataloader = DataLoader(dataset, batch_size=1,
                                     shuffle=False, num_workers=4)

    # dataloader_iterator = iter(dataloader)
    # for i in range(len(dataset)):



    with torch.no_grad():
        for data in dataloader:
            inputs = data['features']
            labels = data['action']
            outputs = model(inputs)
            out1 = model.fc1(inputs)
            out1np = out1.detach().numpy()
            out = torch.relu(out1)
            o = model.out_layer(out)
            # probability_distribution = torch.nn.functional.softmax(outputs)
            prediction = np.argmax(outputs.detach().numpy())
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

    cnn_inference()
    # svm_inference()
