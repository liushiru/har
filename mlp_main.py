import copy
import os
import time
import argparse
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.inspection import permutation_importance

import config
from torch.utils.data import Subset, DataLoader
from captum.attr import FeaturePermutation

from model import MLP
from preprocess import FeatureDataset


def split_dataset(dataset):
    train_len = dataset.train_num
    datasets = {}
    datasets['test'] = Subset(dataset, np.arange(train_len, len(dataset)))

    remaining_data = Subset(dataset, np.arange(train_len))

    train = int(train_len * (1 - config.val_split))
    val = train_len - train
    train_set, val_set = torch.utils.data.random_split(remaining_data,
                                                       [train, val])

    datasets['train'] = train_set
    datasets['val'] = val_set

    return datasets

def get_confusion_matrix(model, dataloader):
    test_y = []
    prediction = []

    for data in dataloader['val']:
        inputs = data['features']
        labels = data['action']
        test_y.extend(labels)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        prediction.extend(predicted)

    cm = sklearn.metrics.confusion_matrix(test_y, prediction)
    return cm


def train_model(model, dataloader, criterion, optimizer):

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(1, config.epochs+1):

        print('Epoch {}/{}'.format(epoch, config.epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for data in dataloader[phase]:
                inputs = data['features']
                labels = data['action']

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader[phase].dataset)
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            # best_model_wts = copy.deepcopy(model.state_dict())


        if phase == 'val':
            for dataset_name in ['train', 'val']:
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in dataloader[dataset_name]:
                        inputs = data['features']
                        labels = data['action']
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print(dataset_name + 'Accuracy: %.2f' % (
                        100 * correct / total))
    return model


if __name__ == "__main__":
    dataset = FeatureDataset(root_dir="Data")
    datasets = split_dataset(dataset)
    dataloader = {}
    dataloader['train'] = DataLoader(datasets['train'], batch_size=config.batch_size,
                                      shuffle=True, num_workers=4)
    dataloader['val'] = DataLoader(datasets['val'], batch_size=config.batch_size,
                                      shuffle=True, num_workers=4)
    dataloader['test'] = DataLoader(datasets['test'], batch_size=config.batch_size,
                                      shuffle=True, num_workers=4)
    model = MLP()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model = train_model(model, dataloader, criterion, optimizer)
    # model_int8 = torch.quantization.quantize_dynamic(
    #     model,  # the original model
    #     {torch.nn.Linear},  # a set of layers to dynamically quantize
    #     dtype=torch.qint8)

    torch.save(model.state_dict(), config.cnn_model_path)


    cm = get_confusion_matrix(model, dataloader)
    print(cm)

    cm_df = pd.DataFrame(cm, index=np.arange(12), columns=np.arange(12))
    cm_df.to_csv('./Data/ConfusionMatrix/mlp_cm.csv')



'''
The following code are attempt to calculate permutation importance
    # model = MLP()
    # model.load_state_dict(torch.load(config.mlp_model_path))
    # X = dataset.get_train_x().astype('float32')
    # X = torch.tensor(X)
    # y = dataset.get_train_y()
    # feature_perm = FeaturePermutation(model.predict)
    # attr = feature_perm.attribute(X, target=y)
'''




