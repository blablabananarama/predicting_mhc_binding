
# coding: utf-8

# In[338]:


import keras


# In[339]:


from keras.models import Sequential
from keras.layers import Dense


# In[340]:


import numpy as np
import matplotlib.pyplot as plt


# In[341]:


np.random.seed(7)


# # Load data

# In[342]:


train_data = "A0201/f000"
valid_data = "A0201/c000"


# # Arguments

# In[343]:


MAX_PEP_SEQ_LEN=9


# # Functions

# In[344]:


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


# In[345]:


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


# In[346]:


def encode_pep(Xin, max_pep_seq_len):
    '''
    encode AA seq of peptides using BLOSUM50
    parameters:
        - Xin : list of peptide sequences in AA
    returns:
        - Xout : encoded peptide seuqneces (batch_size, max_pep_seq_len, n_features)
    '''
    # read encoding matrix:
    blosum = read_blosum_MN('BLOSUM50')
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

# In[347]:


# read in peptide sequences and targets:
X_train_raw, y_train_raw = read_pep(train_data, MAX_PEP_SEQ_LEN)
X_val_raw, y_val_raw = read_pep(valid_data, MAX_PEP_SEQ_LEN)


# ## Encode data

# In[348]:


# encode data using BLOSUM50:
X_train = encode_pep(X_train_raw, MAX_PEP_SEQ_LEN)
y_train = np.array(y_train_raw)
X_val = encode_pep(X_val_raw, MAX_PEP_SEQ_LEN)
y_val = np.array(y_val_raw)


# In[349]:


# data dimensions now:
# (N_SEQS, SEQ_LENGTH, N_FEATURES)
print(X_train.shape)
print(X_val.shape)

n_features = X_train.shape[2]


# In[350]:


# Reshape data
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)


# ## Compile model

# In[363]:


# create model
model = Sequential()
keras.layers.normalization.BatchNormalization(epsilon=1e-06, momentum=0.95, weights=None)
model.add(Dense(20*20, input_dim=MAX_PEP_SEQ_LEN * n_features, activation='relu', init='uniform', kernel_regularizer=keras.regularizers.l2(l=0.1)))
model.add(Dense(50, activation='sigmoid', init='uniform'))
model.add(Dense(10, activation='relu', init='uniform'))
model.add(Dense(1, activation='sigmoid', init='uniform'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'],)


# In[364]:


model.summary()


# ## Fit model

# In[365]:


EPOCHS = 200
MINI_BATCH_SIZE = 50

y_train


# In[366]:


# Fit the model
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=MINI_BATCH_SIZE, validation_data=(X_val, y_val))


# ## Training metrics

# In[355]:


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[356]:


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# ## Evaluate model

# In[358]:


# evaluate the model
scores = model.evaluate(X_val, y_val)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

