import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import config
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from preprocess import FeatureDataset, AverageMeter


def get_confusion_matrix(model, X, y, save=True):
    predict = model.predict(X)
    cm = confusion_matrix(y, predict)
    if save:
        pd.DataFrame(cm, index=np.arange(cm.shape[1]), columns=np.arange(cm.shape[0])).to_csv(config.confusion_matrix_trad_path, mode='a')
    return cm




def k_fold_eval():
    dataset = FeatureDataset(root_dir="Data")
    X, y = dataset.get_data_input()
    skf = StratifiedKFold(n_splits=config.K)
    skf.get_n_splits(X, y)
    accuracy = AverageMeter()

    best_acc = float('-inf')
    count = 0
    for train_index, val_index in skf.split(X, y):
        count += 1
        print('K iteration {}/{}'.format(count, config.K))
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # model = KNeighborsClassifier(n_neighbors=config.num_neighbors, weights='distance', metric='euclidean')
        model = SVC()

        model.fit(X_train, y_train)
        predict = model.predict(X_val)
        acc = np.sum(y_val == predict) / len(y_val)
        accuracy.update(acc)
        print(acc)

        if acc > best_acc:
            best_acc = acc
            get_confusion_matrix(model, X_val, y_val)
    pd.DataFrame({'SVM Acc(partial features)': accuracy.avg}, index=['acc']).to_csv(config.confusion_matrix_trad_path, mode='a')

    return accuracy.val

if __name__ == "__main__":
    dataset = FeatureDataset(root_dir="Data")
    k_fold_eval()

    # train_x, train_y, test_x, test_y = dataset.get_shuffled_dataset()
    #
    # # model = KNeighborsClassifier(n_neighbors=config.num_neighbors, weights='distance', metric='euclidean') #89.2%
    # model = SVC()
    #
    #
    # model.fit(train_x, train_y)
    # predict = model.predict(test_x)
    # accuracy = np.sum(test_y == predict) / len(test_y)
    # print(accuracy)
    #
    # print('training accuracy')
    # train_predict = model.predict(train_x)
    # print((np.sum(train_y == train_predict) / len(train_y)))

    # result = permutation_importance(model, test_x, test_y, n_repeats=1, random_state=0, n_jobs=-1)
    # result_df = pd.DataFrame(index=np.arange(561),columns=['name', 'mean', 'std'])
    # count = 0
    # for i in result.importances_mean.argsort()[::-1]:
    #     result_df.iloc[count, 0] = dataset.features_names[i]
    #     result_df.iloc[count, 1] = result.importances_mean[i]
    #     result_df.iloc[count, 2] = result.importances_std[i]
    #     count=count+1
    # result_df.to_csv('./Data/ConfusionMatrix/svm_importance.csv')