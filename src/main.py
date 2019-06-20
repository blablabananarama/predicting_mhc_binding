from util.i_o import *
from util.fit_model import *
import numpy as np
import matplotlib.pyplot as plt
import os.path as path


data_path = path.abspath(path.join(__file__,"../../dataset/"))

blosum = read_blosum_MN(data_path + "/BLOSUM50")
np.random.seed(0)
len_pep = 9

'''This file contains the i/o as well as the fitting of the models and their validation'''

best_performance = None 
model = None
threshold = 1-np.log(500)/np.log(50000)

current_folder = data_path + "/A1101"
for i in range(0,5):
    tr_file = current_folder + "/f00" + str(i)
    te_file = current_folder + "/c00" + str(i)
    X_train_raw, y_train_raw = read_pep(tr_file, len_pep)
    X_test_raw, y_test_raw = read_pep(te_file, len_pep)
    
    X_train = encode_pep(blosum, X_train_raw, len_pep)
    y_train = np.array(y_train_raw, dtype=float)
    
    for i in range(1, len(y_train)):
        if y_train[i] > threshold:
            y_train[i] = 0
        else:
            y_train[i] = 1


    X_test = encode_pep(blosum, X_test_raw, len_pep)
    y_test = np.array(y_test_raw, dtype=float)
   
    
    for i in range(1, len(y_test)):
        if y_test[i] > threshold:
            y_test[i] = 0
        else:
            y_test[i] = 1



    print(y_test)
    # train_model: takes training and test arrays, gives accuracy score and model object
    
    pcc, mse, auc = train_model(X_train, y_train, X_test, y_test, "DCNN")
    print(mse)
    print(auc)
    if model_performance < best_performance:
        best_performance = mse 
     #   best_model = model
        
    
