from i_o import *
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
len_pep = 9

'''This file contains the i/o as well as the fitting of the models and their validation'''

best_performance = None 
model = None

for i in range(0,5):
    tr_file = "f00" + i
    te_file = "c00" + i
    X_train_raw, y_train_raw = read_pep(tr_file, len_pep)
    X_test_raw, y_test_raw = read_pep(te_file, len_pep)

    X_train = encode_pep(X_train_raw, len_pep)
    y_train = np.array(y_train_raw)

    X_test = encode_pep(X_test_raw, len_pep)
    y_test = np.array(y_val_raw)
    
    # train_model: takes training and test arrays, gives accuracy score and model object
    model_performance, model = train_model(X_train, y_train, X_test, y_test)
    
    if model_performance > best_performance:
        best_performance = model_performance
        best_model = model
        

