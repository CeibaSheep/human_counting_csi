# -*- coding: utf-8 -*-
#==============================================================================
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch.autograd import Variable
import numpy as np
import librosa
#==============================================================================
learning_rate = 0.0005
num_epoches = 50
batch_size = 10 # train 235 *1000 
# test row = 129761
pca_dim = 10
var_batch = 50
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
        # spectral centroid
        # for j in range(batch.shape[1]):
        #     y =  np.ravel((batch[:,j]).T)
        #     sc = librosa.feature.spectral_centroid(y, sr=8000)[0]
        #     batch_sc = np.append(batch_sc,sc)
        # batch_sc = np.array(batch_sc)
        # batch_sc = np.reshape(batch_sc, (1, batch_sc.shape[0]))
        batch_feature = np.hstack((batch_mean, batch_std, batch_var))
        # batch_feature = (batch_feature[0]).tolist()
        batch_feature = np.ravel(batch_feature)
        var_matrix.append(batch_feature)
    var_matrix = np.array(var_matrix)
    var_matrix = var_matrix.reshape([dim, (pca_dim-1)*3])
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
        # spectral centroid
        # for j in range(batch.shape[1]):
        #     y =  np.ravel((batch[:,j]).T)
        #     sc = librosa.feature.spectral_centroid(y, sr=8000)[0]
        #     batch_sc = np.append(batch_sc,sc)
        # batch_sc = np.array(batch_sc)
        # batch_feature = np.hstack((batch_mean, batch_std, batch_var, batch_sc))
        batch_feature = np.hstack((batch_mean, batch_std, batch_var))
        batch_feature = (batch_feature[0]).tolist()
        batch_label = label[i*var_batch:(i+1)*var_batch]
        if (np.var(batch_label) == 0):
            batch_label = label[i * var_batch]
            var_matrix.append(batch_feature)
            resized_label.append(batch_label)
    
    resized_label = torch.tensor(np.array(resized_label))
    var_matrix = np.array(var_matrix)
    var_matrix = torch.tensor(var_matrix.reshape([var_matrix.shape[0], (pca_dim-1)*3]))
    return var_matrix, resized_label

train_dataset = torch.from_numpy(get_X('../X_train.npy'))
train_label = torch.from_numpy(get_Y('../Y_label.npy'))

test_dataset, test_label = get_test_data()
# CUDA_LAUNCH_BLOCKING = 1
input_dim = train_dataset.size(1)

#===================================================================
# define sinple forword neural network

class Neuralnetwork(nn.Module):
    def __init__(self, in_dim, nn_hidden_1, nn_hidden_2, out_dim):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, nn_hidden_1),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(nn_hidden_1, nn_hidden_2),
            nn.Softmax()
        )
        # self.layer3 = nn.Sequential(
        #     nn.Linear(nn_hidden_2, nn_hidden_3),
        #     nn.ReLU(True)
        # )
        # self.layer4 = nn.Sequential(
        #     nn.Linear(nn_hidden_3, out_dim),
        #     nn.Softmax())
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        return x
# ===========================================================
train_len =  int(train_dataset.size(0) / batch_size)
test_len = int(test_dataset.shape[0] / batch_size)

def getbatch(index, X, Y):
    start = index * batch_size
    end = (index + 1) * batch_size
    data = X[start:end, :]
    label = Y[start:end]
    return data, label

model = Neuralnetwork(1*input_dim, 100, 50, 6) # need to be modified
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epoches):
    print ('epoch{}'.format(epoch + 1))
    print ('*'*10)
    running_loss = 0.0
    running_acc = 0.0
    index = 0
    for index in range(train_len):
        x_train, y_train = getbatch(index, train_dataset, train_label)
        if torch.cuda.is_available():
            x_train = Variable(x_train).cuda()
            y_train = Variable(y_train).cuda()
        else :
            x_train = Variable(x_train)
            y_train = Variable(y_train)
        # forward
        out = model.forward(x_train.float())
        # size1, size2 = out.size(0), out.size(1)
        loss = criterion(out, y_train.long())
        running_loss += loss.data * y_train.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == y_train.long()).sum()
        running_acc += num_correct.data

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if index % 100 == 0 :
            print ('{}/{} Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch+1, num_epoches, float(running_loss)/(batch_size*(index+1)), float(running_acc)/(batch_size * (index+1))
            ))
    print ('Finish {} epoch, Loss: {:.6f}, Acc:{:.6f}'.format(
        epoch + 1, float(running_loss) / (len(train_dataset)), float(running_acc) / (len(train_dataset)) 
    ))

    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0

    for index in range(test_len):
        x_test, y_test = getbatch(index, test_dataset, test_label)
        if torch.cuda.is_available():
            x_test = Variable(x_test).cuda()
            y_test = Variable(y_test).cuda()
        else :
            x_test = Variable(x_test)
            y_test = Variable(y_test)
        out = model.forward(x_test.float())
        loss = criterion(out, y_test.long())
        eval_loss += loss.data * y_test.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == y_test.long()).sum()
        eval_acc += num_correct.data
    print ('test loss: {:.6f}, Acc: {:.6f}'.format(
        float(eval_loss) / (len(test_dataset)), float(eval_acc) / (len(test_dataset)))
    )
print ('Done!')
# torch.save(model.state_dict(), './model1.pth')