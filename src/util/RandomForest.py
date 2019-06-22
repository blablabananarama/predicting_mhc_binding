import os
import sys
import math
import random
import sklearn
import warnings
import numpy as np 
from numpy.linalg import inv
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from IPython.core.debugger import set_trace
from util.i_o import *
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

'''
MAX_PEP_SEQ_LEN = 9
max_pep_seq_len = 9

os.chdir("../")
os.chdir("../")
os.chdir("dataset/")

train_data = "A3101/f000"
testid_data = "A3101/c000"
print(os.getcwd())

#Xin = len(X_train_raw)
blosum=read_blosum_MN('BLOSUM50')

# read in peptide sequences and targets:
X_train_raw, y_train_raw = read_pep(train_data, MAX_PEP_SEQ_LEN)
X_test_raw, y_test_raw = read_pep(testid_data, MAX_PEP_SEQ_LEN)

# encode data using BLOSUM50:
X_train = encode_pep(blosum,X_train_raw ,max_pep_seq_len)
y_train = np.array(y_train_raw)
X_test = encode_pep(blosum,X_test_raw ,max_pep_seq_len)
y_test = np.array(y_test_raw)
'''

#%%
def RandomForest(X_train, y_train, X_test, y_test, model_name):
    FINAL_AUC = 0
    print("")
    print("Starting Random Forest...")
    for CHANGE in range(2,11):
        # RESHAPING TRAIN
        nsamples, nx, ny = X_train.shape
        X_train_reshape = X_train.reshape((nsamples,nx*ny))
        
        # FIT FOREST
        clf = RandomForestClassifier(n_estimators=106, min_samples_split=CHANGE, random_state=0)
        clf.fit(X_train_reshape, y_train)
        
        # RESHAPING TEST
        Test_nsamples, Test_nx, Test_ny = X_test.shape
        X_test_reshape = X_test.reshape((Test_nsamples,Test_nx*Test_ny))
          
        # VALIDATION    
        y_test_pred = clf.predict(X_test_reshape)
        
        y_test_pred = y_test_pred.astype(float)
        y_test_pred = y_test_pred.astype(int)
        y_test = y_test.astype(float)
        y_test = y_test.astype(int)
        
        AUC = roc_auc_score(y_test,y_test_pred)
        
        if CHANGE != 1:
            if AUC > FINAL_AUC:
                FINAL_AUC = AUC
                
        if CHANGE == 1:
            FINAL_AUC = AUC
        
        print("Gridsearching:", np.around(CHANGE/11,decimals=2),"%")
        
    return y_test_pred,FINAL_AUC

