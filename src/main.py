from util.i_o import *
from util.fit_model import *
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
from util.DCNN import *








#global discrete = True
#global shuffle = True

data_path = path.abspath(path.join(__file__,"../../dataset/"))

blosum = read_blosum_MN(data_path + "/BLOSUM50")
np.random.seed(0)
len_pep = 9

'''This file contains the i/o as well as the fitting of the models and their validation'''

best_performance = 0 
model = "" 
threshold = 1-np.log(500)/np.log(50000)

tr_folder = "A1101"
current_folder = data_path + "/" + tr_folder
for i in range(0,5):
    # reading in the data of the current fold of training from the training and test files
    cur_file = "f00" + str(i)
    tr_file = current_folder + "/" + cur_file
    te_file = current_folder + "/c00" + str(i)
    X_train_raw, y_train_raw = read_pep(tr_file, len_pep)
    X_test_raw, y_test_raw = read_pep(te_file, len_pep)
   
    # encoding the data with blosum encoding as well as encoding it to binary X_train = encode_pep(blosum, X_train_raw, len_pep)
    X_train = encode_pep(blosum, X_train_raw, len_pep)
    y_train = np.array(y_train_raw, dtype=float)
   
    encode_binary(y_train)

    # encoding the data with blosum as well as encoding it to binary
    X_test = encode_pep(blosum, X_test_raw, len_pep)
    y_test = np.array(y_test_raw, dtype=float)
   
    # 
    encode_binary(y_test) 

    print(y_test)
    # train_model: takes training and test arrays, gives accuracy score and model object
    all_aucs=[]    
    mse, auc, path_to_model= train_model(X_train, y_train, X_test, y_test, "RF", tr_folder + cur_file)
    all_aucs.append(auc)
    mse, auc, path_to_model= train_model(X_train, y_train, X_test, y_test, "ANN", tr_folder + cur_file)
    all_aucs.append(auc)
    mse, auc, path_to_model= train_model(X_train, y_train, X_test, y_test, "DCNN", tr_folder + cur_file)
    all_aucs.append(auc)






    for j in range(0, len(all_aucs)):
        if all_aucs[j] > best_performance:
            best_performance = all_aucs[j] 
            best_model = str(j) + str(i) 
    

    
       
print("Best AUC: ", best_performance)
print("Best Model: ", best_model)

