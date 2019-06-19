from i_o import *
import numpy as np
import matplotlib.pyplot as plt


data_path = "/home/jonas/Documents/algo_project/predicting_mhc_binding/dataset/A0101/"


np.random.seed(0)
len_pep = 9

'''This file contains the i/o as well as the fitting of the models and their validation'''

best_performance = None 
model = None

for i in range(0,5):
    tr_file = data_path + "f00" + str(i)
    te_file = data_path + "c00" + str(i)
    X_train_raw, y_train_raw = read_pep(tr_file, len_pep)
    X_test_raw, y_test_raw = read_pep(te_file, len_pep)

    X_train = encode_pep(X_train_raw, len_pep)
    y_train = np.array(y_train_raw)

    X_test = encode_pep(X_test_raw, len_pep)
    y_test = np.array(y_val_raw)
   
    print(X_test[0])
    # train_model: takes training and test arrays, gives accuracy score and model object
    '''
    model_performance, model = train_model(X_train, y_train, X_test, y_test)
    
    if model_performance > best_performance:
        best_performance = model_performance
        best_model = model
        
    '''
