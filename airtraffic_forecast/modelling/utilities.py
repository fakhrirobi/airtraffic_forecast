#inside utilities.py
#importing all required packages
from statsmodels.tsa.stattools import adfuller #for stationarity test 
from contextlib import contextmanager # for temporary path changer
import numpy as np # numpy operations
import os # for temporary path changer
import pandas as pd 


#function to change os temporary using contextmanager
function to change path whenever loading timeseries data
@contextmanager
def data_load_path(path:str)->None : 
    """ function to change path whenever loading timeseries data

    Args:
        path ([str]): [description]
    
    """
    import os
    prev_cwd = os.getcwd()
    print(os.getcwd())
    os.chdir('..')
    os.chdir(f'data/{path}')
    print(os.getcwd())
    try : 
        yield
    finally : 
        os.chdir(prev_cwd) 
        
@contextmanager
def save_model_path(path='model_output') : 
    """path changer whenever saving the model 

    Args:
        path (str, optional): path of location where the model is saved. Defaults to 'model_output'.
    """
    import os
    prev_cwd = os.getcwd()
    os.chdir(f'{path}')
    try : 
        yield
    finally : 
        os.chdir(prev_cwd)   
        
@contextmanager
def load_model_performance_path(path='model_performance') : 
    """path changer whenever loading the performance of each timeseries method : statistical, ml, NN 
    to create report 

    Args:
        path (str, optional): path of the location of where the model performance result (.csv) is saved . Defaults to 'model_performance'.
    """
    import os
    prev_cwd = os.getcwd()
    os.chdir(f'{path}')
    try : 
        yield
    finally : 
        os.chdir(prev_cwd)          
        
@contextmanager
def save_figure_path(path='model_performance') : 
    """ path changer for saving the plotly figure 

    Args:
        path (str, optional): where to save the figure . Defaults to 'model_performance'.
    """
    import os
    prev_cwd = os.getcwd()
    print(prev_cwd)
    os.chdir(f'{path}')
    try : 
        yield
    finally : 
        os.chdir(prev_cwd)  
         
            
#stationary test 
def stationary_test(data : pd.DataFrame,ts : str ,alpha_threshold: float=0.05)-> bool : 
    """a function that return whether the timeseries data is stationary or not 

    Args:
        data (pd.DataFrame): pandas dataframe of forecast dataset 
        ts (str): time_series feature to be tested stationarity
        alpha_threshold (float, optional): degree of error . Defaults to 0.05.

    Returns:
        [bool]: Return boolen value of True -> if stationary otherwise False
    """
    #
    result = adfuller(data[ts])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    #rejection logic 
    if result[1] > alpha_threshold : 
        print('The Data Is Non Stationary') 
        return False
    else : 
        print('The Data Is Stationary')
        return True





def transform_data(data : pd.DataFrame,ts:string:'Total Passenger')->pd.DataFrame :
    """set of transformation function to make the dataset stationary, 
    

    Args:
        data (pd.DataFrame): pandas dataframe of forecast dataset 
        ts (string): original feature that not stationary 

    Returns:
        pd.DataFrame: return collection of transformation method timeseries data
    """

    data['ts_log'] = data[ts].apply(lambda x: np.log(x))

    #rolling logarithm transformation with 12 month rolling mean 
    data['ts_log_moving_avg'] = data['ts_log'].rolling(window = 12,
                                                                   center = False).mean()

    # moving avg with 12 month windows
    data['ts_moving_avg'] = data[ts].rolling(window = 12,
                                                           center = False).mean()

    #differencing logarithm transformed value 
    data['ts_log_diff'] = data['ts_log'].diff()

    #differencing ts with its average 
    data['ts_moving_avg_diff'] = data[ts] - data['ts_moving_avg']

    # differencing log with its log moving average 
    data['ts_log_moving_avg_diff'] = data['ts_log'] - data['ts_log_moving_avg']


   

    # EMWA transformation 
    data['ts_log_ewma'] = data['ts_log'].ewm(halflife = 12,ignore_na = False, min_periods = 0,adjust = True).mean()




    #differencing EMWA with its log 
    data['ts_log_ewma_diff'] = data['ts_log'] - data['ts_log_ewma']
    #root square 
    data['sqrt_ts'] = np.sqrt(data[ts])
    
    #rolling sqrt 
    data['moving_avg_sqrt'] = data['sqrt_ts'].rolling(window = 12,
                                                                   center = False).mean()
    data['diff_sqrt_moving_avg'] = data['sqrt_ts']-data['moving_avg_sqrt']
    #removing NaN
    data = data.dropna()
    return data







