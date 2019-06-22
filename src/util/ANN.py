# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 14:51:49 2019

@author: Sasha
"""


import os
import keras

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
from keras.models import load_model
from scipy.stats import pearsonr

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = False

#session = tf.Session(config=config)
#set_session(session)



#with tf.device('/gpu:0'):
#  var = tf.Variable(initial_value=([[1,2],[3,4]]))
#sess = tf.Session()
#sess.run(var)

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
# In[30]:


import numpy as np
import matplotlib.pyplot as plt

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
# In[5]:
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

np.random.seed(7)


# # Load data

# In[6]:


train_data = "../../dataset/A0201/f000"
valid_data = "../../dataset/A0201/c000"


# # Arguments

# In[31]:


MAX_PEP_SEQ_LEN=9

print(os.getcwd())


# # Functions

# In[23]:




def read_pep(filename, MAX_PEP_SEQ_LEN):
    '''
    read AA seq of peptides and MHC molecule from text file
    parameters:
        - filename : file in which data is stored
    returns:
        - pep_aa : list of amino acid sequences of peptides (as string)
        - target : list of log transformed IC50 binding values
    '''
    pep_aa = []
    target = []
    infile = open(filename, "r")

    for l in infile:
        l = l.strip().split()
        assert len(l) == 3
        if len(l[0]) <= MAX_PEP_SEQ_LEN:
            pep_aa.append(l[0])
            target.append(l[1])
    infile.close()

    return pep_aa, target


# In[24]:


def read_blosum_MN(filename):
    '''
    read in BLOSUM matrix
    parameters:
        - filename : file containing BLOSUM matrix
    returns:
        - blosum : dictionnary AA -> blosum encoding (as list)
    '''

    # read BLOSUM matrix:
    blosumfile = open(filename, "r")
    blosum = {}
    B_idx = 99
    Z_idx = 99
    star_idx = 99

    for l in blosumfile:
        l = l.strip()

        if l[0] != '#':
            l = l.strip().split()

            if (l[0] == 'A') and (B_idx==99):
                B_idx = l.index('B')
                Z_idx = l.index('Z')
                star_idx = l.index('*')
            else:
                aa = str(l[0])
                if (aa != 'B') &  (aa != 'Z') & (aa != '*'):
                    tmp = l[1:len(l)]
                    # tmp = [float(i) for i in tmp]
                    # get rid of BJZ*:
                    tmp2 = []
                    for i in range(0, len(tmp)):
                        if (i != B_idx) &  (i != Z_idx) & (i != star_idx):
                            tmp2.append(float(tmp[i]))

                    #save in BLOSUM matrix
                    blosum[aa] = tmp2
    blosumfile.close()
    return(blosum)


# In[51]:


def encode_pep(Xin, max_pep_seq_len):
    '''
    encode AA seq of peptides using BLOSUM50
    parameters:
        - Xin : list of peptide sequences in AA
    returns:
        - Xout : encoded peptide seuqneces (batch_size, max_pep_seq_len, n_features)
    '''
    # read encoding matrix:
    blosum = read_blosum_MN("../../dataset/BLOSUM50")
    n_features = len(blosum['A'])
    n_seqs = len(Xin)

    # make variable to store output:
    Xout = np.zeros((n_seqs, max_pep_seq_len, n_features),
                       dtype=np.uint8)

    for i in range(0, len(Xin)):
        for j in range(0, len(Xin[i])):
            Xout[i, j, :n_features] = blosum[ Xin[i][j] ]
    return Xout


# # Main

# ## Read data

# In[40]:


# read in peptide sequences and targets:
X_train_raw, y_train_raw = read_pep(train_data, MAX_PEP_SEQ_LEN)
X_val_raw, y_val_raw = read_pep(valid_data, MAX_PEP_SEQ_LEN)


# ## Encode data

# In[41]:



shuffle = True
discrete = True
# encode data using BLOSUM50:
X_train = encode_pep(X_train_raw, MAX_PEP_SEQ_LEN)
y_train = np.array(y_train_raw)
X_val = encode_pep(X_val_raw, MAX_PEP_SEQ_LEN)
y_val = np.array(y_val_raw)

def ANN(X_train, y_train, X_val, y_val, model_name):
    n_features = X_train.shape[2]
    
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    
    y_train = y_train.astype(float)
    y_train = y_train.reshape(y_train.shape[0], 1)
    
    y_val = y_val.astype(float)
    y_val = y_val.reshape(y_val.shape[0], 1)
    
    
    def disc(ys):
        ys_c = np.empty([len(ys),1])
        for i in range(len(ys)):
            if ys[i] < .425625:
                ys_c[i] = 0
            else: 
                ys_c[i] = 1
        return ys_c
    
    y_train = disc(y_train)
    y_val = disc(y_val)
    
    #y_train_c = categoric(y_train).astype(str)
    #y_val_c = categoric(y_val).astype(str)
    
    
    if shuffle:
        shuf_train = np.arange(len(X_train))
        np.random.shuffle(shuf_train)
        shuf_val = np.arange(len(X_val))
        np.random.shuffle(shuf_val)
        
        X_train = X_train[shuf_train,:]
        y_train = y_train[shuf_train,:]
        #y_train_c = y_train_c[shuf_train,:]
        X_val = X_val[shuf_val,:]
        y_val = y_val[shuf_val,:]
        #y_val_c = y_val_c[shuf_val,:]
        
    
    
    
    # In[42]:
    
    
    # data dimensions now:
    # (N_SEQS, SEQ_LENGTH, N_FEATURES)
    print(X_train.shape)
    print(X_val.shape)



# In[43]:


# Reshape data
    
    # ## Compile model
    
    # In[44]:
    
    
    # create model
    model = Sequential()
    keras.layers.normalization.BatchNormalization(epsilon=1e-06, momentum=0.95, weights=None)
    model.add(Dense(20*20, input_dim=MAX_PEP_SEQ_LEN * n_features, activation='relu', init='random_normal', kernel_regularizer=keras.regularizers.l2(l=0.1)))
    model.add(Dense(200, activation='sigmoid', init='random_normal'))
    model.add(Dense(100, activation='sigmoid', init='random_normal'))
    model.add(Dense(1, activation='sigmoid', init='random_normal'))      # init = "uniform"
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])     # binary_crossentropy , mse
    
    
    # In[45]:
    
    
    model.summary()
    
    
    # ## Fit model
    
    # In[46]:
    
    
    EPOCHS = 100
    MINI_BATCH_SIZE = 120
    
    
    # In[47]:
    
    print(X_train.shape)
    #%%
    
    
    # Fit the model
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=MINI_BATCH_SIZE, validation_data=(X_val, y_val))
    train_loss = history.history["loss"]
    test_loss = history.history["val_loss"]
    model_path = os.path.abspath(os.path.join(__file__,"../../../dataset/models"))
    model_path =  model_path + "/" + model_name + "_" + "model.h5"
    #print(model_path)
    #model_path = "work_bitch.h5"
    model.save(model_path)
    y_pred = model.predict(X_val)
    if discrete:
        y_pred =(y_pred > 0.5)    
    return train_loss, test_loss, y_val, y_pred, model_path #y_real, y_predict, history

a,b,c,d,e = ANN(X_train, y_train, X_val, y_val, "ANN")