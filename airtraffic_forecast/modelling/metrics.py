#inside metrics.py
#importing all required packages 
from sklearn import metrics 
from pprint import pprint
import numpy as np 


def get_metrics(ytrue : np.array,yhat : np.array,dict_output : bool=True) : 
    """
    function that compiled three metrics as output 
    1. RMSE 
    2. MAE 
    3. MAPE 

    Args:
        ytrue (np.array): actual value of y 
        yhat (np.array): forecasted value
        dict_output (bool, optional): [description]. Defaults to True.

    Returns:
        dict if json_output = True , int if json_output = False
    """


    #using sklearn.metrics to calculate the forecast performance 
    rmse_ = metrics.mean_squared_error(ytrue,yhat)
    mae_ = metrics.mean_absolute_error(ytrue,yhat)
    mape_ = metrics.mean_absolute_percentage_error(ytrue,yhat)
    
    if dict_output : 
        metrics_combined = {
            'RMSE Score' : rmse_ , 
            'MAE Score' : mae_ , 
            'MAPE Score' : mape_
        }
        
        print('Model Result : ')
        pprint(metrics_combined)
        
        return metrics_combined
    else : 
        
        
        return rmse_,mae_,mape_