# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 18:42:18 2020

@author: WZM
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
import sklearn.tree as tree
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def data_pre(data):
    data_pre = data
    for cols in data_pre.columns:
        data_pre.loc[:,cols] = list(map(lambda x: max(0,x),data_pre.loc[:,cols]))
    return data_pre

def data_basic(data):
    data_pre = data
    for cols in data_pre.columns:
        data_pre.loc[data_pre[cols] == -99,cols] = np.mean(data_pre[data_pre[cols] != -99][cols])
    return data_pre

def data_m(data):
    data_pre = data
    data_af = pd.DataFrame()
    for i in list(set(data_pre.id)):
        print(i)
        data_i = data_pre[data_pre.id == i]
        data_i.index = range(len(data_i.index))
        for cols in data_i.columns[data_i[data_i == -99].any()].tolist():
            for k in data_i.index:
                if data_i.loc[k,cols] == -99:
                    if k == 0:
                        data_i.loc[k,cols] = 0
                    else:
                         data_i.loc[k,cols] = data_i.loc[k-1,cols]
        for cols in data_i.columns[2:]:
            value = 0.8*data_i.loc[:,cols][-1:] + 0.2*np.mean(data_i.loc[:,cols][:-1])
            data_af.loc[i,cols] = float(value)
            data_af.loc[i,cols+'_mean'] = np.mean(data_i.loc[:,cols])
            data_af.loc[i,cols+'std'] = np.std(data_i.loc[:,cols])
    return data_af
            
    

def m_data(data):
    data_af = pd.DataFrame()
    for i in list(set(data.id)):
        print(i)
        data_i = data[data.id == i]
        for cols in data_i.columns[2:]:
            value = 0.7*data_i.loc[:,cols][-1:] + 0.3*np.mean(data_i.loc[:,cols][:-1]) 
            data_af.loc[i,cols] = float(value)
    return data_af

def model_train(x,y,X_test):
    x_train,x_test,y_train,y_test = train_test_split(x,y)
    x_train = x_train.iloc[:,1:]
    y_train = y_train.iloc[:,1]
    x_test = x_test.iloc[:,1:]
    y_test = y_test.iloc[:,1]
    X_test = X_test.iloc[:,1:]
    #model = svm.SVC(probability=True)
    #model = LogisticRegression(class_weight = 'balanced',solver = 'sag')
    #model = tree.DecisionTreeClassifier(criterion='entropy',max_depth = 8,min_samples_split = 5)
    #model = MLPClassifier(solver='adam', activation='logistic',alpha=1e-8,tol=1e-10,hidden_layer_sizes=(50,50), random_state=1,max_iter=100,verbose=10,learning_rate_init=.1)
    
    lgb_train = lgb.Dataset(x_train,y_train)
    params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': { 'auc'},
    'num_leaves': 36,
    'learning_rate': 0.01,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.7,
    'random_state':1024,
    }
    model = lgb.train(params,  lgb_train,  num_boost_round=950)
    
    
    #model.fit(x,y)
    y_test_predict = model.predict(x_test)
    score = roc_auc_score(y_test, y_test_predict)
    print(score)
    return score

def predict_y(x,y,X_test):
    x_train = x.iloc[:,1:]
    y_train = y.iloc[:,1:]
    X_test = X_test.iloc[:,1:]
    lgb_train = lgb.Dataset(x_train,y_train)
    params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': { 'auc','binary_logloss'},
    'num_leaves': 18,
    'learning_rate': 0.01,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.7,
    'random_state':1024,
    }
    model = lgb.train(params,  lgb_train,  num_boost_round=950)
    predict = model.predict(X_test)
    return predict

def test(n):
    auc = 0
    for i in range(n):
        auc = auc + model_train(x_train,y_train,x_test)
    return auc/n

'''
data_basic_test = data_basic(pd.read_csv('data_b_test.csv'))
data_m_test = data_m(pd.read_csv('data_m_test.csv'))
x_test = pd.concat([data_basic_test,data_m_test],axis = 1)
x_test.to_csv('x_test.csv')
x_train = pd.read_csv('x_train2.csv')
y_train = pd.read_csv('y_train.csv')
'''

'''
data_basic_train = data_basic(pd.read_csv('data_b_train.csv'))
data_basic_test = data_basic(pd.read_csv('data_b_test.csv'))

data_m_train = data_m(pd.read_csv('data_m_train.csv'))
data_m_test = data_m(pd.read_csv('data_m_test.csv'))
y_train = pd.read_csv('y_train.csv')

x_test = pd.concat([data_basic_test,data_m_test],axis = 1)
x_train = pd.concat([data_basic_train,data_m_train],axis = 1)


y_train = pd.read_csv('y_train.csv')
'''
x_train = pd.read_csv('x_train2.csv')
x_train = x_train.iloc[:,1:]
x_test = pd.read_csv('x_test.csv')
x_test = x_test.iloc[:,1:]
y_train = pd.read_csv('y_train.csv')
predict = pd.DataFrame(index = x_test.id) 
predict['predict'] = predict_y(x_train,y_train,x_test)
predict.to_csv('predict.csv')
 