# -*- coding: utf-8 -*-
#==============================================================================
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
#==============================================================================
learning_rate = 0.001
num_epoches = 100
batch_size = 10 # train 235 *1000 
# test row = 129761
pca_dim = 30
var_batch =150
#========================================================================
def zeroMean(data):
    meanVal = np.mean(data, axis=0)
    newData = data - meanVal
    return newData, meanVal

def pca(data, n = pca_dim):
    newData, meanVal = zeroMean(data)
    covMat = np.cov(newData, rowvar=0)

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(n+1):-1]
    n_eigVect = eigVects[:, n_eigValIndice]
    lowDDataMat = newData * n_eigVect
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal
    lowDDataMat = np.delete(lowDDataMat, 0, axis= 1)
    reconMat = np.delete (reconMat, 0, axis=1)
    return lowDDataMat, reconMat

def get_X(path):
    data = np.load(path)
    data = data[:,0:56]
    lowDDdata, reconMat = pca(data)
    var_matrix = []
    dim = round(lowDDdata.shape[0]/var_batch)
    batch_sc = []
    for i in range(dim):
        batch = lowDDdata[i*var_batch:(i+1)*var_batch]
        batch_std = np.std(batch, axis = 0)
        batch_var = np.var(batch, axis = 0)
        batch_mean = np.mean(batch, axis = 0)
        batch_feature = np.hstack((batch_mean, batch_std, batch_var))
        batch_feature = np.ravel(batch_feature)
        var_matrix.append(batch_feature)
    var_matrix = np.array(var_matrix)
    return var_matrix

def get_Y(path):
    label = np.load(path)
    label = label.T
    label = label[0]
    resized_label = []
    dim = round(label.size/ var_batch)
    for i in range(dim):
        batch_label = label[i*var_batch]
        resized_label.append(batch_label)
    resized_label = np.array(resized_label)
    # label = torch.from_numpy(label[0])
    return resized_label

def get_test_data():
    data = np.load('../X_test.npy')
    data = data[:,0:56]
    lowDDdata, reconMat = pca(data)
    var_matrix = []

    label = np.load('../Y_test_label.npy')
    label = label.T
    label = label[0]
    resized_label = []
    dim = round(label.size/ var_batch)

    for i in range(dim):
        batch = lowDDdata[i*var_batch:(i+1)*var_batch]
        batch_std = np.std(batch, axis = 0)
        batch_var = np.var(batch, axis = 0)
        batch_mean = np.mean(batch, axis = 0)
        batch_feature = np.hstack((batch_mean, batch_std, batch_var))
        batch_feature = np.ravel(batch_feature)
        batch_label = label[i*var_batch:(i+1)*var_batch]
        if (np.var(batch_label) == 0):
            batch_label = label[i * var_batch]
            var_matrix.append(batch_feature)
            resized_label.append(batch_label)
    
    resized_label = np.array(resized_label)
    var_matrix = np.array(var_matrix)
    return var_matrix, resized_label

train_dataset = get_X('../X_train.npy')
train_label = get_Y('../Y_label.npy')

test_dataset, test_label = get_test_data()
#===================================================================
# SVM multi-classes
x = train_dataset
y = train_label
clf_rbf = svm.SVC(kernel='sigmoid', gamma='scale', decision_function_shape='ovo').fit(x, y)


test_pred = clf_rbf.predict(test_dataset)
test_label = pd.DataFrame(test_label)
union_actual_pred = pd.concat([test_label,pd.DataFrame(test_pred)],axis = 1)


recall = union_actual_pred[union_actual_pred.iloc[:,0]==1][union_actual_pred.iloc[:,1]==1].count()/union_actual_pred[union_actual_pred.iloc[:,0]==1].count()
percison = union_actual_pred[union_actual_pred.iloc[:,0]==1][union_actual_pred.iloc[:,1]==1].count() / union_actual_pred[union_actual_pred.iloc[:,1]==1].count()
correction = union_actual_pred[union_actual_pred.iloc[:,0]==union_actual_pred.iloc[:,1]].count()/union_actual_pred.iloc[:,0].count()

print ('about the linear svm , the recall is %s' %recall)
print ('about the linear svm , the percison is %s' %percison)
print ('about the linear svm , the correction is %s' %correction)
print ('Done!')
# torch.save(model.state_dict(), './model1.pth')