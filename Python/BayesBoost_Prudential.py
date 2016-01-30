# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:35:19 2016

@author: ivanliu
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from xgboost import XGBClassifier
import xgboost as xgb
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
from bayes_opt import BayesianOptimization

DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH = '../data/test.csv'

def load_data(path_train = DATA_TRAIN_PATH, path_test = DATA_TEST_PATH):
    columns_to_drop = ['Id', 'Response', 'Medical_History_1']

    print("Load the data using pandas")
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    
    # combine train and test
    all_data = train.append(test)
    
    # Found at https://www.kaggle.com/marcellonegro/prudential-life-insurance-assessment/xgb-offset0501/run/137585/code
    # create any new variables    
    all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[1]
    all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[2]
    
    # factorize categorical variables
    all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
    all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
    all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]
    
    print('Eliminate missing values')    
    # Use -1 for any others
    all_data.fillna(-1, inplace=True)
    
    # fix the dtype on the label column
    all_data['Response'] = all_data['Response'].astype(int)
    
    # split train and test
    train = all_data[all_data['Response']>0].copy()
    test = all_data[all_data['Response']<1].copy()
    labels = train['Response']
    xgtrain = train.drop(columns_to_drop, axis=1)
    xgtest = test.drop(columns_to_drop, axis=1)
    #xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), train['Response'].values)
    #xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response'].values)    

    return xgtrain, xgtest, labels
    
def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              gamma,
              min_child_weight,
              max_delta_step,
              subsample,
              colsample_bytree,
              silent =True,
              nthread = -1,
              seed = 1234):
    return cross_val_score(XGBClassifier(max_depth = int(max_depth),
                                         learning_rate = learning_rate,
                                         n_estimators = int(n_estimators),
                                         silent = silent,
                                         nthread = nthread,
                                         gamma = gamma,
                                         min_child_weight = min_child_weight,
                                         max_delta_step = max_delta_step,
                                         subsample = subsample,
                                         colsample_bytree = colsample_bytree,
                                         seed = seed,
                                         objective = "multi:softprob"),
                           train,
                           labels,
                           "log_loss",
                           cv=5).mean()   

train, test, labels = load_data()
xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (5, 10),
                                      'learning_rate': (0.01, 0.3),
                                      'n_estimators': (50, 1000),
                                      'gamma': (1., 0.01),
                                      'min_child_weight': (2, 10),
                                      'max_delta_step': (0, 0.1),
                                      'subsample': (0.7, 0.8),
                                      'colsample_bytree' :(0.5, 0.99)
                                     })

xgboostBO.maximize()
print('-'*53)

print('Final Results')
print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])