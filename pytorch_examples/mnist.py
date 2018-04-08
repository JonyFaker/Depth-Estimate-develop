import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

## some Variable for training setting
use_cuda = False

batch_size = 4
learning_rate = 0.01
momentum = 0.5
epochs = 4

mytransform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
trandataset = datasets.MNIST('./data', train=True, download=True, transform=mytransform)
testdataset = datasets.MNIST('./data', train=False, transform=mytransform)

train_loader = torch.utils.data.DataLoader(trandataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=True)

class Net(nn.Module):
    """docstring for Net."""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
if use_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

def train(epoch):
    model.train()
    for i_batch, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if i_batch % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i_batch * len(data), len(train_loader.dataset),
            100. * i_batch / len(train_loader), loss.data[0]
            ))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
        ))

for epoch in range(1, epochs + 1):
    train(epoch)
    test()
