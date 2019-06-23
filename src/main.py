import seaborn as sns; sns.set()
from util.i_o import *
from util.fit_model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as path
from util.DCNN import *
from scipy.stats import pearsonr
from os import listdir
import statistics

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


all_folders = listdir(data_path)
all_folders = list(filter(lambda a: a != 'models' and a != 'BLOSUM50', all_folders))
data_exception = []

average_auc_RF_total = 0
average_auc_ANN_total = 0
average_auc_DCNN_total = 0

average_cc_DCNN_total = 0
average_cc_RF_total = 0
average_cc_ANN_total = 0

all_cc_RF = []
all_cc_ANN = []
all_cc_DCNN = []

all_auc_RF =[]
all_auc_ANN = []
all_auc_DCNN = []

for folder in all_folders:
    tr_folder = folder
    current_folder = data_path + "/" + tr_folder
    RF_auc_list, RF_cc_list = [],[]
    ANN_auc_list, ANN_cc_list = [],[]
    DCNN_auc_list, DCNN_cc_list = [],[] 
    discrete = False

    for i in range(0,2):
        # reading in the data of the current fold of training from the training and test files
        try: 
            cur_file = "f00" + str(i)
            tr_file = current_folder + "/" + cur_file
            te_file = current_folder + "/c00" + str(i)
            X_train_raw, y_train_raw = read_pep(tr_file, len_pep)
            X_test_raw, y_test_raw = read_pep(te_file, len_pep)
           
            # encoding the data with blosum encoding as well as encoding it to binary X_train = encode_pep(blosum, X_train_raw, len_pep)
            X_train = encode_pep(blosum, X_train_raw, len_pep)
            y_train = np.array(y_train_raw, dtype=float)
           


            # encoding the data with blosum as well as encoding it to binary
            X_test = encode_pep(blosum, X_test_raw, len_pep)
            y_test = np.array(y_test_raw, dtype=float)

            if discrete:
                encode_binary(y_train)
                encode_binary(y_test) 

            # train_model: takes training and test arrays, gives accuracy score and model object
                all_aucs=[]
                
                t_test, y_pred, path_to_model= train_model(X_train, y_train, X_test, y_test, "RF", tr_folder + cur_file)
                mse = mean_squared_error(t_test, y_pred)
                auc = roc_auc_score(t_test, y_pred)        
                all_aucs.append(auc)
                RF_auc_list.append(auc)
                
                t_test, y_pred, path_to_model= train_model(X_train, y_train, X_test, y_test, "ANN", tr_folder + cur_file)
                y_pred =(y_pred > 0.5)
                mse = mean_squared_error(t_test, y_pred)
                auc = roc_auc_score(t_test, y_pred)
                all_aucs.append(auc)
                ANN_auc_list.append(auc)
                
                t_test,y_pred, path_to_model= train_model(X_train, y_train, X_test, y_test, "DCNN", tr_folder + cur_file)
                mse = mean_squared_error(t_test, y_pred)
                auc = roc_auc_score(t_test, y_pred)
                all_aucs.append(auc)
                DCNN_auc_list.append(auc)
            
                for j in range(0, len(all_aucs)):
                    if all_aucs[j] > best_performance:
                        best_performance = all_aucs[j] 
                        best_model_auc = str(j) + str(i) 
                

            else: 
                all_ccs=[]
                
                
                #mse, auc, y_pred, path_to_model= train_model(X_train, y_train, X_test, y_test, "RF", tr_folder + cur_file)
                #corr, p_value = pearsonr(y_test, y_pred)
                #all_ccs.append(corr)
                #RF_cc_list.append(corr)
                
                t_test, y_pred, path_to_model= train_model(X_train, y_train, X_test, y_test, "ANN", tr_folder + cur_file)
                corr, p_value = pearsonr(t_test, y_pred)
                print("THIS IS CC: {}".format(corr))
                all_ccs.append(corr)
                ANN_cc_list.append(corr)
                
                t_test, y_pred, path_to_model= train_model(X_train, y_train, X_test, y_test, "DCNN", tr_folder + cur_file)
                print(t_test)
                print(y_pred)
                t_test = t_test.reshape(len(t_test),1)
                corr, p_value = pearsonr(t_test, y_pred)
                all_ccs.append(corr)
                DCNN_cc_list.append(corr)
                
                
                for j in range(0, len(all_ccs)):
                    if all_ccs[j] > best_performance:
                        best_performance = all_ccs[j] 
                        best_model_cc = str(j) + str(i)
            
        
            if discrete == False:
                 all_cc_RF.append(np.mean(RF_cc_list))
                 all_cc_ANN.append(np.mean(ANN_cc_list))
                 all_cc_DCNN.append(np.mean(DCNN_cc_list))
            else:
                all_auc_RF.append(np.mean(RF_auc_list))
                all_auc_ANN.append(np.mean(ANN_auc_list))
                all_auc_DCNN.appendi(np.mean(DCNN_auc_list))

            print(all_cc_DCNN)
            print(all_cc_RF)
            print(all_cc_ANN)
        except Exception as e:
            data_exception.append(folder)
            print(folder)
            print(e)

#%% 
#Rf_Avg = np.average(RF_list)
#Ann_Avg = np.average(ANN_list)
#Dcnn_Avg = np.average(DCNN_list)
#All_Auc_Avg = np.array([Rf_Avg, Ann_Avg, Dcnn_Avg])

#%%
# COMPUTE PLOT
#import pandas as pd
#All_Auc_Avg = pd.DataFrame(All_Auc_Avg).T
#All_Auc_Avg.columns = ['RF','ANN','DCNN']
#print(All_Auc_Avg)

#%%
#sns.set(color_codes=True)
#plt.bar(All_Auc_Avg['RF'],'C9',label='Random Forest')
#plt.bar(All_Auc_Avg['ANN'],'C9',label='ANN')
#plt.bar(All_Auc_Avg['DCNN'],'C9',label='DCNN')

#plt.figure(figsize=(13, 7))
#ax = sns.barplot(y=All_Auc_Avg)
#    plt.plot(TrainLoss,label="Train Loss")
#    plt.legend(fontsize='x-large', title_fontsize=40)
#plt.xlabel("", fontsize=20)
#plt.ylabel("Loss", fontsize=20)
            
print("mean cc RF" ,np.mean(all_cc_RF))
print("mean cc ANN",np.mean(all_cc_ANN))
print("mean cc DCNN",np.mean(all_cc_DCNN))
print("mean auc RF", np.mean(all_auc_RF))
print("mean auc ANN", np.mean(all_auc_ANN))
print("mean auc DCNN",np.mean(all_auc_DCNN))

    
       
print("Best AUC: ", best_performance)
print("Best Model: ", best_model)

