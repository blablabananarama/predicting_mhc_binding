from util.DCNN import Amazing_DCNN
from scipy.stats import pearsonr 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score


def train_model(X_train, y_train, X_test, y_test, model_type, model_name):
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
        y_train_loss, y_test_loss, t_test, y_pred, path_to_model = Amazing_DCNN(X_train, y_train, X_test, y_test, model_name) 
    elif(model_type == "PEN"):
        pass
    
    # train the network, predict the output
   # compute the auc of the model 
    pcc = 2
    #pcc = pearsonr(t_test, y_pred)
    mse = mean_squared_error(t_test, y_pred)
    auc = roc_auc_score(t_test, y_pred)
    print(t_test)
    print(y_pred)
    return(pcc, mse, auc, path_to_model)
    
