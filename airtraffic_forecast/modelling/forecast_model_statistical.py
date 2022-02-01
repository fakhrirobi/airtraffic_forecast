import itertools


import numpy as np 
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")



from statsmodels.tsa.statespace.sarimax import SARIMAX
from contextlib import contextmanager
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
    #//TODO : add list of default parameter 
    def __init__(self) : 
        pass 
    
    def train_model(self,data,ts,save_model=False,custom_name=None,p=1,d=0,q=1,P=1,D=0,Q=1,s=12)->None : 
        #//TODO : refer to init some variables 
        data_ = data.copy()
        data_.drop('Period',axis=1,inplace=True)
        data_ = data_[[ts]]
        TEST_LENGTH = 29 
        TRAIN_LENGTH = 100 
        rmse_compiled = []
        mae_compiled = []
        mape_compiled = []
        for i in range(1,TEST_LENGTH) : 
            TRAIN_DATA = data_.loc[:TRAIN_LENGTH,:]
            TEST_DATA = data_.loc[TRAIN_LENGTH:,:]
            
            model = SARIMAX(TRAIN_DATA, order=(p,d,q),
                            seasonal_order=(P,D,Q,s),initialization='approximate_diffuse')
            #changing package from statsmodels to sktime 
            fitted_model = model.fit()
            predicted_values = fitted_model.forecast(steps=TEST_LENGTH)
            rmse_,mae_,mape_ = get_metrics(ytrue=TEST_DATA,yhat=predicted_values,json_output=False)
            # append all the result 
            rmse_compiled.append(rmse_)
            mae_compiled.append(mae_)
            mape_compiled.append(mape_)
            TRAIN_LENGTH +=1
            TEST_LENGTH -= 1
            
    
        model_name_format = f'SARIMA_{p},{d},{q}_{P},{D},{Q},{s}_{ts}'
        model_performance = {
            'RMSE Score' : np.mean(rmse_compiled) , 
            'MAE Score' : np.mean(mae_compiled) , 
            'MAPE Score' : np.mean(mape_compiled)
        }
        
        model_performance['model_name'] = model_name_format
        model_performance_df = pd.DataFrame(model_performance,index=[0])
        

        
        if save_model : 
            
            filename_format = f'SARIMA_{p},{d},{q}_{P},{D},{Q},{s}__{ts}.pkl'
            data_train = data.copy()
            
            data_train.set_index('Period',inplace=True)
            data_train = data_train[[ts]]
            model = SARIMAX(data_train, order=(p,d,q),
                            seasonal_order=(P,D,Q,s),initialization='approximate_diffuse')
            model_fit = model.fit()
            with path_changer() : 
                model_fit.save(filename_format)
   
        return model_performance_df
    
    
    def bic_selection_optimal(self,ts_train)-> pd.DataFrame(): 
        # from the model above on SARIMA we are going to use the moving avg sqrt 
        p_min = 0
        d_min = 0
        q_min = 0
        p_max = 10
        d_max = 0
        q_max = 10

        # Initialize a DataFrame to store the results
        results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                                columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])

        for p,d,q in itertools.product(range(p_min,p_max+1),
                                    range(d_min,d_max+1),
                                    range(q_min,q_max+1)):
            if p==0 and d==0 and q==0:
                results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
                continue

            try:
                model = SARIMAX(ts_train, order=(p, d, q),
                                    #enforce_stationarity=False,
                                    #enforce_invertibility=False,
                                    )
                results = model.fit()
                results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
            except:
                continue
        results_bic = results_bic[results_bic.columns].astype(float)
        return results_bic


# model finding







# class HoltWinterModel : 
#     def __init__(self) -> None:
#         pass
        
        
    