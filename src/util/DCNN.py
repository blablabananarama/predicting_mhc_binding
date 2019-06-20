import os
import sys
import math
import random
import sklearn
import warnings
import torch.optim
import numpy as np 
from numpy.linalg import inv
warnings.filterwarnings("ignore")
from numpy.linalg import multi_dot as mdot
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import seaborn as sns; sns.set()
from IPython.core.debugger import set_trace

from i_o import *
MAX_PEP_SEQ_LEN = 9
max_pep_seq_len = 9

###  NAVIGATE
os.chdir("../")
os.chdir("../")
os.chdir("dataset/")

train_data = "A0201/f000"
testid_data = "A0201/c000"

#Xin = len(X_train_raw)
blosum=read_blosum_MN('BLOSUM50')

# read in peptide sequences and targets:
X_train_raw, y_train_raw = read_pep(train_data, MAX_PEP_SEQ_LEN)
X_test_raw, y_test_raw = read_pep(testid_data, MAX_PEP_SEQ_LEN)

# encode data using BLOSUM50:
X_train = encode_pep(blosum,X_train_raw ,max_pep_seq_len).astype(float)
y_train = np.array(y_train_raw).astype(float)
X_test = encode_pep(blosum,X_test_raw ,max_pep_seq_len).astype(float)
y_test = np.array(y_test_raw).astype(float)

X_train = np.transpose(X_train,(0,2,1))
X_test = np.transpose(X_test,(0,2,1))

#%%
def Amazing_DCNN(X_train, y_train, X_test, y_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 50
    train_dl = DataLoader(X_train, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dl = DataLoader(y_test, batch_size=batch_size, shuffle=False, num_workers=4)
    tensor = torch.Tensor([batch_size,21,9])
    tensor = tensor.float()
    print("tensor:",tensor)
    
        #### ARCHITECTURE ####
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            self.drop_out = nn.Dropout(0.7)
            self.conv1 = nn.Conv1d(in_channels=21, out_channels=44, kernel_size=2, padding=0)
            self.fc1 = nn.Linear(in_features=(352), out_features=110, bias =True)
            self.fc2 = nn.Linear(in_features=(110), out_features=1, bias =True)
    
            ### FORWARD FUNCTION WITH DROP IN EVERY LAYER EXCEPT OUTPUT-LAYER   		
        def forward(self,tensor):
            out_t = F.relu(self.drop_out(self.conv1(tensor)))  
            out_t = out_t.view(out_t.size(0),-1)
            out_t = F.softsign(self.drop_out(self.fc1(out_t)))
            out_t = torch.sigmoid(self.drop_out(self.fc2(out_t)))
            out_t = torch.reshape(out_t,(len(out_t),1))
            return out_t   
   
    model = Network().to(device)
    model = model.float()
    num_epochs = 20
    learning_rate = 0.00005
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Preparing Training Loop
    print("Starting Deep Convolutional Neural Network...")

    TrainLoss = [] # List
    TestLoss = [] # List
    EpochList = [] # List
    
    for epoch in range(num_epochs):
        model.train()
        EpochList.append(1+len(EpochList))
        for i, data in enumerate(train_dl.batch_sampler):    
            # Zero the optimizer
            optimizer.zero_grad()    
            # GRAB INFORMATION
            seqs = torch.from_numpy(X_train[data,:,:]) 
            labels = torch.from_numpy(y_train[data])
            #FORWARD PASS
            outputs = model(seqs.float())
            loss = criterion(outputs,labels.float())   
            #BACKPROP & ADAM OPTIMIZATION     
            loss.backward(loss)
            optimizer.step()
        TrainLoss.append(loss.item())
        print(TrainLoss[-1])  
        
        model.eval()
        with torch.no_grad():
            for i,data in enumerate(test_dl.batch_sampler):
                test_seqs = torch.from_numpy(X_test[data,:,:])
                test_labels = torch.from_numpy(y_test[data])
                y_ = model(test_seqs.float())  
                
            TestLoss.append(criterion(y_,test_labels.float()).item())
        
        
    print("Epoch:",len(EpochList))
    print("Train:",len(TrainLoss))         
    print("Test:",len(TestLoss)) 
    
    sns.set(color_codes=True)
    plt.figure(figsize=(13, 7))
    ax = sns.lineplot(x=EpochList, y=TestLoss,label="Test Loss").set_title('Test Loss', fontsize=20)
    plt.plot(TrainLoss,label="Train Loss")
    plt.legend(fontsize='x-large', title_fontsize=40)
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    
    y_pred = model(torch.from_numpy(X_test[:,:,:]).float())
    
    return TrainLoss[-1], TestLoss[-1], y_test, y_pred 
#%%
print(Amazing_DCNN(X_train, y_train, X_test, y_test))

#%%
