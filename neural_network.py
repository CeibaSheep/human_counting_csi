import torch
from torch import nn, optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch.autograd import Variable

batch_size = 32
learning_rate = 1e-2
num_epoches = 50

# down load MNIST hand-write dataset

train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor()
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

# define sinple forward neural network
class Neuralnetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim),
            nn.ReLU(True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
model = Neuralnetwork(28 * 28, 300, 100, 10)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    print ('epoch{}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available() :
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else :
            img = Variable(img)
            label = Variable(label)
        # go forward
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        # print('num_correct', num_correct.data)
        running_acc += num_correct.data

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%300 == 0 :
            print ('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch+1, num_epoches, running_loss/(batch_size * i), float(running_acc)/(batch_size * i)
            ))
    print ('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), float(running_acc) / (len(train_dataset))
    ))
    model.eval()
    eval_loss = 0
    eval_acc = 0
     
    for data in test_loader:
        img, label = data
        img = img.view(img.size(), -1)
        if torch.cuda.is_available() :
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else :
            img = Variable(img)
            label = Variale(label)
        
        out =  model(img)
        loss = criterion(out, label)
        eval_loss += loss.data * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data

    print ('Test Loss: {:.6f}, Acc: {:.6f}'.format(
        eval_loss/(len(test_dataset)), float(eval_acc)/(len(test_dataset))
    ))

#save model
torch.save(model.state_dict(), './neural_network.pth')
                # for epoch in range(num_epoches):
              
