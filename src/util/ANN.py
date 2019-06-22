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

import numpy as np
import matplotlib.pyplot as plt


#shuffle = True
#discrete = True


def ANN(X_train, y_train, X_val, y_val, model_name):
    
    EPOCHS = 100
    MINI_BATCH_SIZE = 120
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
#    print(X_train.shape)
 #   print(X_val.shape)

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

#    print(X_train.shape)
    
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

#a,b,c,d,e = ANN(X_train, y_train, X_val, y_val, "ANN")