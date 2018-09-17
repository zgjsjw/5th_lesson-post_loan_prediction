__author__ = 'jw5775'
coding = 'utf-8'

import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, confusion_matrix

data_pti_path = 'C:/Users\JIWei\Desktop\zyed\data_new'
data_pto_path = 'C:/Users\JIWei\Desktop\zyed\data_new'


def pre_data(path):
    train_file_0 = pd.read_csv(os.path.join(path, 'Train_data_0.csv'))
    train_file_1 = pd.read_csv(os.path.join(path, 'Train_data_1.csv'))
    train_file = pd.concat([train_file_0, train_file_1])
    #print(train_file.apply(lambda t:t.isnull().sum()))
    train_data = train_file.fillna(train_file.mean())
    #train_data = train_file.fillna(0)
    #print(train_data.apply(lambda t: t.isnull().sum()))

    test_file_0 = pd.read_csv(os.path.join(path, 'Test_data_0.csv'))
    test_file_1 = pd.read_csv(os.path.join(path, 'Test_data_1.csv'))
    test_file = pd.concat([test_file_0, test_file_1])
    test_data = test_file.fillna(test_file.mean())
    #test_data = test_file.fillna(0)

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
    print(xgb)

    paras = {'max_depth':range(1,3),
             'min_child_weight':[i/10 for i in range(0,10)],
             'scale_pos_weight':range(10,100,10)}
    gscv = GridSearchCV(estimator=xgb, param_grid=paras, cv=5, scoring='roc_auc')
    gscv.fit(x_train, y_train)
    print(gscv.best_params_)
    print(gscv.best_score_)
    print(gscv.score(x_test, y_test))
    result = gscv.predict(x_test)
    print(confusion_matrix(y_test, result))
    print(classification_report(y_test, result))

    xgb.fit(x_train, y_train)
    test_result = xgb.predict(x_test)
    print(confusion_matrix(y_test, test_result))
    print(classification_report(y_test, test_result))
    '''
    result = xgb.score(x_test, y_test)
    print(result)

    pre = precision_score(y_test, xgb.predict(x_test), pos_label=1)
    print(pre)
    rec = recall_score(y_test, xgb.predict(x_test), pos_label=1)
    print(rec)
    acc = accuracy_score(y_test, xgb.predict(x_test))
    print(acc)
    '''


def main():
    train_data, test_data = pre_data(data_pti_path)
    train_model(train_data, test_data)


if __name__ == "__main__":
    main()
