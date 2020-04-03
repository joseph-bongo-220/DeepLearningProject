#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import time


# In[2]:


#custom Dataset class 
class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    def __len__(self):
        return len(self.data)


# In[3]:


numpy_data = np.load('Images/final/final_images.npy')
#reshape the data to be consistent with most workflows, ie (1, 512, 512) instead of (512, 512, 1)
numpy_data = numpy_data.reshape(numpy_data.shape[0], 1, numpy_data.shape[1], numpy_data.shape[2]) 
numpy_target = np.load('Images/final/final_labels.npy')


# In[4]:


#a mediocre train test split
#using 75 points in training set, 25 in test set for toy example
sample = random.sample(range(800), 800)
train_sample = sample[:75]
test_sample = sample[75:100]
train_data = numpy_data[train_sample]
train_target = numpy_target[train_sample]
test_data = numpy_data[test_sample]
test_target = numpy_data[test_sample]


# In[5]:


train_set = MyDataset(train_data, train_target)
test_set = MyDataset(test_data, test_target)


# In[6]:


#Training
n_training_samples = 75
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

#Test
n_test_samples = 25
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))


# In[7]:


class CNN(torch.nn.Module):
    
    #Our batch shape for input x is (1, 512, 512)
    
    def __init__(self):
        super(CNN, self).__init__()
        
        #Input channels = 1, output channels = 16
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(16 * 16 * 16, 64)
        
        #64 input features, 10 output features for our 2 defined classes
        self.fc2 = torch.nn.Linear(64, 2)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (1, 512, 512) to (16, 512, 512)
        x = F.relu(self.conv1(x))
        
        #Size changes from (16, 512, 512) to (18, 256, 256)
        x = self.pool1(x)
        #Size changes from (16, 256, 256) to (16, 128, 128)
        x = self.pool2(x)
        #Size changes from (16, 128, 128) to (16, 64, 64)
        x = self.pool3(x)        
        #Size changes from (16, 64, 64) to (16, 32, 32)
        x = self.pool4(x)   
        #Size changes from (16, 64, 64) to (16, 16, 16)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (16, 16, 16) to (1, 4096)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 4096)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 2)
        x = self.fc2(x)
        
        return(x)


# In[8]:


def outputSize(in_size, kernel_size, stride, padding):
    
    #helpful when filling in pooling layers
    
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)
outputSize(in_size = 32, kernel_size = 3, stride = 2, padding = 1)


# In[9]:


def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=2)
    return(train_loader)


# In[10]:


test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, sampler=test_sampler, num_workers=2)


# In[11]:


def createLossAndOptimizer(net, learning_rate=0.001):
    
    #Loss function
    loss = torch.nn.CrossEntropyLoss()
    
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    return(loss, optimizer)


# In[12]:


def trainNet(net, batch_size, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    
    #Time for printing
    training_start_time = time.time()
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader, 0):
            
            #Get inputs
            inputs, labels = data
            
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


# In[13]:


#create CNN instance
CNN = CNN()
#train it
trainNet(CNN, batch_size=1, n_epochs=10, learning_rate=0.001)


# In[ ]:




