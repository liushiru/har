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
from preprocess import FeatureDataset, RawDataset, AverageMeter


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


def k_fold_eval(dataset, model_name='CNN'):

    max_acc = float('-inf')
    accuracy = AverageMeter()
    loss = AverageMeter()

    dummy_x = dataset.dataframe[['label']].to_numpy()
    dummy_y = dummy_x.flatten()

    skf = StratifiedKFold(n_splits=config.K)
    skf.get_n_splits(dummy_x, dummy_y)

    count = 0
    for train_index, val_index in skf.split(dummy_x, dummy_y):
        count += 1
        print('K iteration {}/{}'.format(count, config.K))
        dataloaders = get_dataloaders(dataset, train_index, val_index)



        if model_name[:3] == 'CNN':
            model = CNN()
        if model_name[:3] == 'MLP':
            model = MLP()
        if model_name[-2:] == 'LD':
            model.load_state_dict(torch.load(config.model_path))

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        best_model, best_loss, best_acc = train_model(model, dataloaders, criterion, optimizer)

        accuracy.update(best_acc, 1)
        loss.update(best_loss, 1)

        if best_acc > max_acc:
            torch.save(model.state_dict(), config.model_path)
            get_confusion_matrix(best_model, dataloaders, save=True)

    print("K fold eval: acc: {}, loss: {}".format(accuracy, loss))

    # acc_df = pd.DataFrame({'Acc': accuracy.avg},index=['acc']).to_csv(config.confusion_matrix_path, mode='a')
    return accuracy.val, loss.val


def get_dataloaders(dataset, train_index=None, val_index=None):
    if train_index is None:
        return DataLoader(dataset, batch_size=config.batch_size,
                                   shuffle=True, num_workers=4)

    datasets = {}
    datasets['train'] = Subset(dataset, train_index)
    datasets['val'] = Subset(dataset, val_index)
    dataloader = {}
    dataloader['train'] = DataLoader(datasets['train'], batch_size=config.batch_size,
                                     shuffle=True, num_workers=4)
    dataloader['val'] = DataLoader(datasets['val'], batch_size=config.batch_size,
                                   shuffle=True, num_workers=4)
    return dataloader


def get_confusion_matrix(model, dataloader, save=True):
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
    if save:
        pd.DataFrame(cm, index=np.arange(cm.shape[1]), columns=np.arange(cm.shape[0])).to_csv(config.confusion_matrix_path, mode='a')
        # cm.to_csv(config.confusion_matrix_path)

    return cm

def get_permutation_importance():
    importance = pd.read_csv('./Data/ConfusionMatrix/feature_importance.csv', index_col=0)
    feature_names = pd.read_csv(FeatureDataset.feature_names_path, names=['feature_names']).to_numpy().flatten()
    importance.index = feature_names
    importance['abs_val'] = abs(importance.to_numpy())
    sorted_importance = importance.sort_values(by=['abs_val'], ascending=False)
    sorted_importance.to_csv('./Data/ConfusionMatrix/sorted_importance.csv')




def train_model(model, dataloader, criterion, optimizer):
    best_loss = float('inf')
    best_acc = -1
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
            torch.save(model.state_dict(), config.model_path)

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

                accuracy = correct / total
                print(dataset_name + 'Accuracy: %.2f' % (
                        100 * accuracy))

            best_acc = max(accuracy, best_acc)


    model.load_state_dict(torch.load(config.model_path))
    print(model)
    return model, best_loss, best_acc



def load_model():
    model = torch.load(config.model_path)

    for key, val in model.items():
        print(key)
        # npy = val.detach().numpy()
        # pass

        filepath = os.path.join("Data", "ModelWeights", "{}.npy".format(key))
        f = open(filepath, "w")
        with open(filepath, 'wb') as f:
            np.save(f, val.detach().numpy())



if __name__ == "__main__":

    load_model()

    if config.model_name[:3] == 'CNN':
        dataset = RawDataset()
    if config.model_name[:3] == 'MLP':
        dataset = FeatureDataset()

    k_fold_eval(dataset, config.model_name)

    model = CNN()

    model.load_state_dict(torch.load(config.model_path))


    get_confusion_matrix(model)


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


    feature_perm = FeaturePermutation(model)
    for data in dataloader['train']:
        inputs = data['features']
        labels = data['action']
        attr = feature_perm.attribute(inputs, target=labels)









