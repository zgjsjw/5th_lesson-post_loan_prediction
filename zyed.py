__author__ = 'jw5775'
coding = 'utf-8'

import pandas as pd
import os
from sklearn import preprocessing
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

data_in_path = 'C:/Users\JIWei\Desktop\zyed\data_in'
data_out_path = 'C:/Users\JIWei\Desktop\zyed\data_out'


def pre_data(path):
    train_file_0 = pd.read_csv(os.path.join(path, 'Train_data_0.csv'))
    train_file_1 = pd.read_csv(os.path.join(path, 'Train_data_1.csv'))
    train_file = pd.concat([train_file_0, train_file_1])
    train_data = train_file.fillna(train_file.mean())

    test_file_0 = pd.read_csv(os.path.join(path, 'Test_data_0.csv'))
    test_file_1 = pd.read_csv(os.path.join(path, 'Test_data_1.csv'))
    test_file = pd.concat([test_file_0, test_file_1])
    test_data = test_file.fillna(test_file.mean())

    return train_data, test_data

    '''
   file_re_dup = raw_file.drop_duplicates()
   print(file_re_dup.shape)
   return raw_file
   '''


def get_fea_lab(data):
    label = data[data.columns[0]].values
    feature_tmp = data[data.columns[1:]].values
    feature = preprocessing.StandardScaler().fit_transform(feature_tmp)
    return feature, label


def train_model(train, test):
    x_train, y_train = get_fea_lab(train)
    x_test, y_test = get_fea_lab(test)
    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)
    result = xgb.score(x_test, y_test)
    print(result)

    pre = precision_score(y_test, xgb.predict(x_test), pos_label=1)
    print(pre)
    rec = recall_score(y_test, xgb.predict(x_test), pos_label=1)
    print(rec)
    acc = accuracy_score(y_test, xgb.predict(x_test))
    print(acc)


def main():
    train_data, test_data = pre_data(data_in_path)
    train_model(train_data, test_data)


if __name__ == "__main__":
    main()
