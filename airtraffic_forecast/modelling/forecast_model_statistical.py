#inside forecast_model_statistical.py 
#importing all required packages 
import itertools # for hyperparameter finding for order of ARMA component ( p and q )

#common operations
import numpy as np 
import pandas as pd
import os




from statsmodels.tsa.statespace.sarimax import SARIMAX # SARIMAX Model 
from contextlib import contextmanager # for path changer
from datetime import datetime


#importing metrics module 
from metrics import get_metrics


@contextmanager
def path_changer(path='model_output') : 
    current_wd = os.getcwd()
    os.chdir(path)
    try : 
        yield 
    finally : 
        os.chdir(current_wd)
        
class SarimaxModel : 
    """
    SarimaxModel Class the purpose of creating this class instead of direct implementation using statsmodels is to provide 
    time series crossvalidation 
    """
    
    def train_model(self,data : pd.DataFrame,ts:str,save_model: bool=False,custom_name=None,p=1,d=0,q=1,P=1,D=0,Q=1,s=12)->None : 
        """ train the SARIMAX model 

        Args:
            data (pd.DataFrame): timeseries data for training 
            ts (str): [description] : feature name  / columns of training dataset -> because there are no single ts (due to transformation)
            save_model (bool, optional): if True model will be saved to .pkl format  Defaults to False.
            custom_name ([type], optional): [description]. Defaults to None.
            p (int, optional): AR order. Defaults to 1.
            d (int, optional): Differencing order. Defaults to 0.
            q (int, optional): MA order. Defaults to 1.
            P (int, optional): Seasonal AR order . Defaults to 1.
            D (int, optional): Seasonal differencing order. Defaults to 0.
            Q (int, optional): Seasonal MA order. Defaults to 1.
            s (int, optional): Seasonality . Defaults to 12.

        Returns:
            [pd.DataFrame]: recap of the model result 
        """
       
        data_ = data.copy()
        #drop period columns
        data_.drop('Period',axis=1,inplace=True)
        data_ = data_[[ts]]
        #init the TEST and TRAIN length for cross validation 
        TEST_LENGTH = 29 
        TRAIN_LENGTH = 100 
        #empty list as container of each crossval 
        rmse_compiled = []
        mae_compiled = []
        mape_compiled = []
        for i in range(0,TEST_LENGTH) : 
            #change indexing for each fold 
            TRAIN_DATA = data_.loc[:TRAIN_LENGTH,:]
            TEST_DATA = data_.loc[TRAIN_LENGTH:,:]
            
            model = SARIMAX(TRAIN_DATA, order=(p,d,q),
                            seasonal_order=(P,D,Q,s),initialization='approximate_diffuse')
            #fitting on different index for each crossval 
            fitted_model = model.fit()
            # forecast on test length 
            predicted_values = fitted_model.forecast(steps=TEST_LENGTH)
            #return metrics from metrics.py 
            rmse_,mae_,mape_ = get_metrics(ytrue=TEST_DATA,yhat=predicted_values,dict_output=False)
            # append all the result 
            rmse_compiled.append(rmse_)
            mae_compiled.append(mae_)
            mape_compiled.append(mape_)
            #increment train while decreasing the test length 
            TRAIN_LENGTH += 1
            TEST_LENGTH -= 1
            
        #for model saving 
        model_name_format = f'SARIMA_{p},{d},{q}_{P},{D},{Q},{s}_{ts}'
        model_performance = {
            'RMSE Score' : np.mean(rmse_compiled) , 
            'MAE Score' : np.mean(mae_compiled) , 
            'MAPE Score' : np.mean(mape_compiled)
        }
        #adding model name 
        model_performance['model_name'] = model_name_format
        #conver
        model_performance_df = pd.DataFrame(model_performance,index=[0])
        

        
        if save_model : 
            
            filename_format = f'SARIMA_{p},{d},{q}_{P},{D},{Q},{s}__{ts}.pkl'
            data_train = data.copy(deep=True)
            #setting index for datetime feature
            data_train.set_index('Period',inplace=True)
            data_train = data_train[[ts]]
            #fitting the model again for model save 
            model = SARIMAX(data_train, order=(p,d,q),
                            seasonal_order=(P,D,Q,s),initialization='approximate_diffuse')
            model_fit = model.fit()
            with path_changer() : 
                model_fit.save(filename_format)
   
        return model_performance_df
    
    



        
    