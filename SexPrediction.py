#!/usr/bin/env python
# coding: utf-8

# In[106]:


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.transforms as T
import numpy as np
import h5py


# In[107]:


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, train):
        super(EEGDataset).__init__()
        assert x.shape[0] == y.size
        self.x = x
        #temp_y = np.zeros((y.size, 2))
        #for i in range(y.size):
        #    temp_y[i, y[i]] = 1
        #self.y = temp_y
        self.y = [y[i][0] for i in range(y.size)]
        self.train = train
        
    def __getitem__(self,key):
        return (self.x[key], self.y[key])
    
    def __len__(self):
        return len(self.y)


# In[108]:


# Load EEG data
transform = T.Compose([
                T.ToTensor()
            ])
f = h5py.File('child_mind_x_train_v2.mat', 'r')
x_train = f['X_train']
x_train = np.reshape(x_train,(-1,1,24,256))
print('X_train shape: ' + str(x_train.shape))
f = h5py.File('child_mind_y_train_v2.mat', 'r')
y_train = f['Y_train']
print('Y_train shape: ' + str(y_train.shape))
train_data = EEGDataset(x_train, y_train, True)
loader_train = DataLoader(train_data, batch_size=64)

f = h5py.File('child_mind_x_val_v2.mat', 'r')
x_val = f['X_val']
x_val = np.reshape(x_val,(-1,1,24,256))
print('X_val shape: ' + str(x_val.shape))
f = h5py.File('child_mind_y_val_v2.mat', 'r')
y_val = f['Y_val']
print('Y_val shape: ' + str(y_val.shape))
val_data = EEGDataset(x_val, y_val, True)
loader_val = DataLoader(val_data, batch_size=64)

f = h5py.File('child_mind_x_test_v2.mat', 'r')
x_test = f['X_test']
x_test = np.reshape(x_test,(-1,1,24,256))
print('X_test shape: ' + str(x_test.shape))
f = h5py.File('child_mind_y_test_v2.mat', 'r')
y_test = f['Y_test']
print('Y_test shape: ' + str(y_test.shape))
test_data = EEGDataset(x_test, y_test, False)
loader_test = DataLoader(test_data, batch_size=64)


# In[ ]:


print(np.histogram(y_train))


# # Test with MNIST
# import torchvision.datasets as dset
# NUM_TRAIN = 700
# transform = T.Compose([
#                 T.ToTensor(),
#                 T.CenterCrop(24),
#                 T.Pad((116,0))
#             ])
# mnist_train = dset.MNIST('./mnist', train=True, download=True,
#                              transform=transform)
# loader_train = DataLoader(mnist_train, batch_size=64, 
#                           sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
# 
# mnist_val = dset.MNIST('./mnist', train=True, download=True,
#                            transform=transform)
# loader_val = DataLoader(mnist_val, batch_size=64, 
#                         sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
# 
# mnist_test = dset.MNIST('./mnist', train=False, download=True, 
#                             transform=transform)
# loader_test = DataLoader(mnist_test, batch_size=64)

# In[4]:


USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)


# In[5]:


def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


# In[6]:


def train(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(loader_val, model)
                print()


# In[7]:


model = nn.Sequential(
                      nn.Conv2d(1,100,3),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.Dropout(0.25),
                      nn.Conv2d(100,100,3),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.Dropout(0.25),
                      nn.Conv2d(100,300,(2,3)),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.Dropout(0.25),
                      nn.Conv2d(300,300,(1,7)),
                      nn.ReLU(),
                      nn.MaxPool2d((1,2), stride=1),
                      nn.Dropout(0.25),
                      nn.Conv2d(300,100,(1,3)),
                      nn.Conv2d(100,100,(1,3)),
                      nn.Flatten(),
                      nn.Linear(1900,6144),
                      nn.Linear(6144,2),
)

pred = model(next(iter(loader_train))[0])


# In[8]:


print(pred.shape)


# In[9]:


optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
train(model, optimizer)


# 

# In[14]:


best_model = model
check_accuracy(loader_test, best_model)


# In[ ]:




