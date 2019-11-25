import numpy as np
#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.autograd import Variable
#import torch.backends.cudnn as cudnn
import time
import os
import math
#import torchvision
#from torchvision import datasets, models, transforms
from os.path import dirname, join as pjoin
import scipy.io as sio
#
# read data and formulation
datapath = './data/train/'
files = os.listdir(datapath)
files.sort() # a:0-5 phase:0-5
# print (files)
phase = []
amplitude = []
y_label = []
row = 40000*5+35000
X_train = []
# hstack from row; vstack from col
for idx, file in enumerate(files):
    path = os.path.join(datapath, file)
    if idx >= 6 : # store phase
        data = sio.loadmat(path, mdict=None, appendmat=True)
        data = data['train']
        data = data[:, 4:61]
        data = np.delete(data, 28, axis = 1)
        phase = np.append(phase, data)
    else:
        data = sio.loadmat(path, mdict = None, appendmat = True)
        data = data ['train']
        data = np.delete(data, 28, axis = 1)
        amplitude = np.append(amplitude, data)
        label = np.ones(np.size(data, 0))*idx
        y_label = np.append(y_label, label)
#phase = np.array(phase)
phase = np.array(np.split(phase, row))
#amplitude = np.array(amplitude)
amplitude = np.array(np.split(np.array(amplitude), row))
#y_label = np.array(y_label)
y_label = np.array(np.split(y_label, row))

X_train = np.array(np.hstack((amplitude, phase)))
np.save('X_train.npy', X_train)
np.save('Y_label.npy', y_label)

 

# pca


# generate network


# train by GPU
