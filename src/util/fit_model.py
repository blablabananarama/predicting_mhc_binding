from util.DCNN import Amazing_DCNN
from util.ANN import ANN
from util.RandomForest import RandomForest
from scipy.stats import pearsonr 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score


def train_model(X_train, y_train, X_test, y_test, model_type, model_name, shuffle=True):
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
    elif(model_type == "ANN"):
        y_train_loss, y_test_loss, t_test, y_pred, path_to_model = ANN(X_train, y_train, X_test, y_test, model_name) 
    elif(model_type == "RF"):
        y_pred, auc  = RandomForest(X_train, y_train, X_test, y_test, model_name) 
        path_to_model = model_name
        t_test = y_test
        # train the network, predict the output
   # compute the auc of the model 
    mse = mean_squared_error(t_test, y_pred)
    auc = roc_auc_score(t_test, y_pred)
    return(mse, auc, path_to_model)
    
