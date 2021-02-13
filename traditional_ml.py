import pandas as pd
import numpy as np
import config
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from preprocess import HarDataset


if __name__ == "__main__":
    dataset = HarDataset(root_dir="Data")
    train_x = dataset.get_train_x()
    train_y = dataset.get_train_y()
    test_x = dataset.get_test_x()
    test_y = dataset.get_test_y()

    model = KNeighborsClassifier(n_neighbors=config.num_neighbors, weights='distance', metric='euclidean') #89.2%
    model = SVC()#93.6%


    model.fit(train_x, train_y)
    predict = model.predict(test_x)
    accuracy = np.sum(test_y == predict) / len(test_y)
    print(accuracy)

    print('training accuracy')
    train_predict = model.predict(train_x)
    print((np.sum(train_y == train_predict) / len(train_y)))
