# IMPORT MODULE (DONE - RUN TO CHECK)
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np 
from numpy.linalg import inv
from numpy.linalg import multi_dot as mdot
import sklearn
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Enable gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1 LOAD DATA & CHOOSE DATA
# 2 Where to store model?
# 3 PREPARE DATA
# 4 SPLIT TEST TRAIN
# 5 SET BATCH SIZE
# 6 INPUT BUILD DATA CUBE
# 7 SET DATA LOADER
# 8 SET HYPERPARAMETERS
# 9 BUILD ARCHITECTURE

# LOAD HERE

MODEL_STORE_PATH = os.getcwd()
test_size = 0.30
batch_size = 30
train, test = train_test_split( YOUR DATAFRAME , test_size=test_size, random_state=27)

#### BUILD TENSOR ####
NumChan = len(np.transpose(MasterList))
tensor = torch.Tensor([batch_size,NumChan,NumChan*MaxSeq]) # BATCH
tensor = tensor.float()

# INPUT TENSOR
print("")
print("")
print("Your input-tensor dimensions:")
print(tensor) # BatchSize, Number of Channels,  Longest Seq

        ## BUILD TRAINING CUBE - USE DSTACK ##

### DATALOADER
train_dl = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
test_dl = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)

print("")
print("Your dataset:")
print("Batch size:",batch_size)
print("Test size:",np.around((test_size)*100),"%")
print("Train size:",np.around((1-test_size)*100),"%")


### HYPERPARAMETER ###
model = Network().to(device)
model = model.float()
num_epochs = 10
learning_rate = 0.0003
criterion = torch.nn.MSE()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("")
print("Hyper parameters: OK")

#### ARCHITECTURE ####
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.drop_out = nn.Dropout(0.7)
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=44, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(in_features=(1144), out_features=2000, bias =True)
        self.fc3 = nn.Linear(in_features=2000, out_features=60, bias =True)  

        ### FORWARD FUNCTION WITH DROP IN EVERY LAYER EXCEPT OUTPUT-LAYER
		
    def forward(self,tensor):    
        out_t = F.relu(self.drop_out(self.conv1(tensor)))   
        out_t = out_t.view(out_t.size(0),-1)
        apple = torch.tanh(self.drop_out(self.fc1(out_t)))
        out_t = F.relu(self.drop_out(self.fc1(out_t)))
        out_t = torch.sigmoid(self.fc3(out_t * apple))
        out_t = torch.reshape(out_t,(len(out_t),2,MaxSeq))

        return out_t   




# In[10]:

#### TRAINING #### 
model.train() # Set model to train-mode
LossList = [] # Empty list to track loss
TestLoss = [] # Empty list to track test loss
EpochList = []

print("")
print("")
print("Starting Convolutional Neural Network...")
for epoch in range(num_epochs):
     
    model.train()
          
    for i, data in enumerate(train_dl.batch_sampler):   
        # Zero the optimizer
        optimizer.zero_grad()

        # GRAB INFORMATION
        seqs = torch.from_numpy( YOUR TRANING INPUT ) #INPUT
        labels = torch.from_numpy( YOUR TRAINING LABELS ) #LABEL

        #FORWARD PASS
        outputs = model(seqs.float())
        loss = criterion(outputs,labels.float())   
        
        #BACKPROP & ADAM OPTIMIZATION     
        loss.backward(loss)
        optimizer.step()
    LossList.append(loss.item())
    
    #### TESTING ####
    model.eval() #IMPORTANT SET TO EVALUATION MODE
    tempLoss = []
	
    with torch.no_grad():
        for i,data in enumerate(test_dl.batch_sampler):
            test_seqs = torch.from_numpy( YOUR TEST INPUT )
            test_labels = torch.from_numpy( YOUR TEST LABELS )
            y_ = model(test_seqs.float())
            tempLoss.append(criterion(y_,test_labels.float()).item())
    
    TestLoss.append(sum(tempLoss)/len(tempLoss))
    
    
#%%


print()
sns.set(color_codes=True)
plt.figure(figsize=(12, 6))
ax = sns.lineplot(x=EpochList, y=TestLoss,label="Test Loss").set_title('DCNN LOSS', fontsize=16)
plt.plot(LossList,label="Train Loss")
plt.legend(fontsize='x-large', title_fontsize=40)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)

plt.savefig("TrainTestLoss.png")

