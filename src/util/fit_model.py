from DCNN import *
from sklearn.metrics import roc_auc_score




def train_model(X_train, y_train, X_test, y_test, model_type):
    '''
    This function trains different types of model specified 
     - X_train: training data set
     - y_train: training data targets
     - X_test: test data set
     - y_test: test data targets
     - model_type: string of the model model type (DCNN or similar)
    Returns:
     - auc_score: roc auc score computed on the output loss
     - model: fitted model object
    '''

    if (model_type == "DCNN"):
        y_train_loss, y_test_loss = Amazing_DCNN(X_train, y_train, X_test, y_test) 
    elif(model_type == "PEN"):
        pass

    y_train_loss, y_test_loss, model = Amazing_DCNN(X_train, y_train, X_test, y_test) 
   # compute the auc of the model 
    auc_score = roc_auc_score(y_test, y_test_loss)
     

    return(auc_score, model)
