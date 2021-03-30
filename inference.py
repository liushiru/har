import pickle

import torch
import numpy as np
from torch.utils.data import DataLoader

import config
from cnn_main import get_dataloaders, get_confusion_matrix
from model import CNN, MLP
from preprocess import RawDataset, FeatureDataset


def mlp_inference():
    model = MLP()
    model.load_state_dict(torch.load(config.inference_model_path))
    model.eval()

    dataset = FeatureDataset()
    dataloader = DataLoader(dataset, batch_size=1,
                                     shuffle=False, num_workers=4)

    counter = 0
    index = 0
    with torch.no_grad():
        for data in dataloader:
            index += 1
            inputs = data['features']
            labels = data['action']
            outputs = model(inputs)
            # # probability_distribution = torch.nn.functional.softmax(outputs)
            prediction = np.argmax(outputs.detach().numpy())
            # print('prediction of MLP model is {}'.format(prediction))
            # print('label is {}'.format(labels.detach().numpy()[0]))
            # print('----')
            if labels.detach().numpy()[0] != prediction:
                counter += 1
                print(index)
                print('prediction of MLP model is {}'.format(prediction))
                print('label is {}'.format(labels.detach().numpy()[0]))
                print('----')
    print(counter)

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

    mlp_inference()
    # svm_inference()
    model = MLP()
    model.load_state_dict(torch.load(config.inference_model_path))
    model.eval()

    dataset = FeatureDataset()
    dataloader = DataLoader(dataset, batch_size=1,
                                     shuffle=False, num_workers=4)
    dataloaders = {}
    dataloaders['val'] = dataloader
    get_confusion_matrix(model, dataloaders, save=True)
