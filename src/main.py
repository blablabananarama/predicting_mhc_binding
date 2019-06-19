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


current_folder = data_path + "/A0201"
for i in range(0,5):
    tr_file = current_folder + "/f00" + str(i)
    te_file = current_folder + "/c00" + str(i)
    X_train_raw, y_train_raw = read_pep(tr_file, len_pep)
    X_test_raw, y_test_raw = read_pep(te_file, len_pep)

    X_train = encode_pep(blosum, X_train_raw, len_pep)
    y_train = np.array(y_train_raw)

    X_test = encode_pep(blosum, X_test_raw, len_pep)
    y_test = np.array(y_test_raw)
   
    print(X_test[0])
    train_model: takes training and test arrays, gives accuracy score and model object
    
    model_performance, model = train_model(X_train, y_train, X_test, y_test)
    
    model_performance = 1
    best_model = "b"
    if model_performance > best_performance:
        best_performance = model_performance
        best_model = model
        
    
