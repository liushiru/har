import copy
import os
import time
import argparse
import numpy as np
import pandas as pd
import sklearn
import torch

import config
from torch.utils.data import Subset, DataLoader
from captum.attr import FeaturePermutation
from sklearn.model_selection import StratifiedKFold

from model import MLP, CNN2d, CNN
from preprocess import FeatureDataset, RawDataset


def split_dataset(dataset):
    total_len = len(dataset)
    train_len = int(total_len * (1 - config.val_split))
    val_len = total_len - train_len
    train_set, val_set = torch.utils.data.random_split(dataset,
                                                       [train_len, val_len])
    datasets = {}
    datasets['train'] = train_set
    datasets['val'] = val_set

    return datasets


def k_fold_eval(dataset):
    dummy_x = dataset.dataframe[['user_id']].to_numpy()
    dummy_y = dummy_x.flatten()

    skf = StratifiedKFold(n_splits=config.num_users)
    skf.get_n_splits(dummy_x, dummy_y)

    for train_index, val_index in skf.split(dummy_x, dummy_y):
        print('k iter:')
        datasets = {}
        datasets['train'] = Subset(dataset, train_index)
        datasets['val'] = Subset(dataset, val_index)
        dataloader = {}
        dataloader['train'] = DataLoader(datasets['train'], batch_size=config.batch_size,
                                         shuffle=True, num_workers=4)
        dataloader['val'] = DataLoader(datasets['val'], batch_size=config.batch_size,
                                       shuffle=True, num_workers=4)

        model = CNN()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        train_model(model, dataloader, criterion, optimizer)




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

def get_permutation_importance():
    importance = pd.read_csv('./Data/ConfusionMatrix/feature_importance.csv', index_col=0)
    feature_names = pd.read_csv(FeatureDataset.feature_names_path, names=['feature_names']).to_numpy().flatten()
    importance.index = feature_names
    importance['abs_val'] = abs(importance.to_numpy())
    sorted_importance = importance.sort_values(by=['abs_val'], ascending=False)
    sorted_importance.to_csv('./Data/ConfusionMatrix/sorted_importance.csv')




def train_model(model, dataloader, criterion, optimizer):

    for epoch in range(1, config.epochs+1):

        print('Epoch {}/{}'.format(epoch, config.epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()


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
                torch.save(model.state_dict(), config.cnn_model_path)
    return model






if __name__ == "__main__":
    dataset = RawDataset()

    k_fold_eval(dataset)



    datasets = split_dataset(dataset)
    dataloader = {}
    dataloader['train'] = DataLoader(datasets['train'], batch_size=config.batch_size,
                                      shuffle=True, num_workers=4)
    dataloader['val'] = DataLoader(datasets['val'], batch_size=config.batch_size,
                                      shuffle=True, num_workers=4)
    model = CNN()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model = train_model(model, dataloader, criterion, optimizer)

    torch.save(model.state_dict(), config.model_path)


    model.load_state_dict(torch.load(config.model_path))

    model = torch.load(config.mlp_model_path)

    feature_perm = FeaturePermutation(model)
    for data in dataloader['train']:
        inputs = data['features']
        labels = data['action']
        attr = feature_perm.attribute(inputs, target=labels)









