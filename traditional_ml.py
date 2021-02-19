import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

import config
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from preprocess import FeatureDataset


if __name__ == "__main__":
    dataset = FeatureDataset(root_dir="Data")
    # train_x = dataset.get_train_x()
    # train_y = dataset.get_train_y()
    # test_x = dataset.get_test_x()
    # test_y = dataset.get_test_y()

    train_x, train_y, test_x, test_y = dataset.get_shuffled_dataset()

    model = KNeighborsClassifier(n_neighbors=config.num_neighbors, weights='distance', metric='euclidean') #89.2%
    model = SVC()#93.6%


    model.fit(train_x, train_y)
    predict = model.predict(test_x)
    accuracy = np.sum(test_y == predict) / len(test_y)
    print(accuracy)

    print('training accuracy')
    train_predict = model.predict(train_x)
    print((np.sum(train_y == train_predict) / len(train_y)))

    result = permutation_importance(model, train_x, train_y, n_repeats=1, random_state=0)
    result_df = pd.DataFrame(index=np.arange(561),columns=['name', 'mean', 'std'])
    count = 0
    for i in result.importances_mean.argsort()[::-1]:
        result_df.iloc[count, 'name'] = dataset.features_names[i]
        result_df.iloc[count, 'mean'] = result.importances_mean[i]
        result_df.iloc[count, 'std'] = result.importances_std[i]
    result_df.to_csv('./Data/importance.csv')