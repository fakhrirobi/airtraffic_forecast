from copyreg import pickle
import pandas as pd 
import numpy as np
import os 
# TODO : implement pydantic modelling 


#preparing the data 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn import ensemble,linear_model
from tqdm import tqdm
from metrics import get_metrics
from utilities import save_model_path
import joblib, pickle
from tqdm import tqdm 
from math import ceil



    
MODEL_OUTPUT_PATH = 'airtraffic_forecast/modelling/model_output'

class MLForecastingModel() : 
    """
    
    Time Series Forecasting with Machine Learning Algorithm Approach 
    
    """
    
    

    def get_scaler(self,scaler):
        """
        

        Args:
            scaler (str): the name of sklearn scaler 
            scalers :
                "minmax" -> MinMaxScaler,
                "standard"->StandardScaler,
                "maxabs"->MaxAbsScaler,
                "robust"->RobustScaler,
        
        Returns:
            sklearn scaler object
        """
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())

    
    def generate_feature(self,data,ts,lag_num) : 
        """[summary]

        Args:
            data ([pd.Dataframe,pd.Series,np.array]): data for timeseries forecasting
            ts ([str]): [description]
            lag_num ([int]): number of lag for creating feature 

        Returns:
            [pd.Dataframe]
        """
        data = data.copy()
        data['month'] = data.Period.dt.month
        data['year'] = data.Period.dt.year
        for idx in range(1,lag_num+1) : 
            data[f'lag_{idx}'] = data[ts].shift(idx)
        data = data.fillna(0)
        
        return data 

    def create_cyclical_feature(self,data, col_name, period, start_num=0):
        """[summary]

        Args:
            data ([pd.Dataframe]): [description]
            col_name ([str]): [description]
            period ([int]): Month / Annual Period number 
            start_num (int, optional): count start on each period. Defaults to 0.

        Returns:
            [type]: [description]
        """
        kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(data[col_name]-start_num)/period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(data[col_name]-start_num)/period)    
             }
        return data.assign(**kwargs).drop(columns=[col_name])

    def start_prepare_data(self,data,ts='Total Passenger',lag_num=129) :
        """[summary]

        Args:
            data ([type]): [description]
            ts (str, optional): [description]. Defaults to 'Total Passenger'.
            lag_num (int, optional): [description]. Defaults to 129.

        Returns:
            [type]: [description]
        """
        
        data_ts_feature = self.generate_feature(data=data,ts=ts,lag_num=lag_num)
        data_month_cyclical = self.create_cyclical_feature(data=data_ts_feature,col_name='month',period=12,start_num=1)
        data_year_cylical  = self.create_cyclical_feature(data=data_month_cyclical,col_name='year',period=1,start_num=2005)
        #drop the period column = 
        final_data = data_year_cylical.drop('Period',axis=1)
        
        X,y  = self.train_test_split(data=final_data,target_col='Total Passenger')
        
        
        return X,y  
    
    def train_test_split(self,data,target_col,split=False) : 
        """[summary]

        Args:
            data ([type]): [description]
            target_col ([type]): [description]
            split (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """

        scaler = self.get_scaler('minmax')
        X = data.drop(target_col,axis=1).values
        X = scaler().fit_transform(X)
        y = data.loc[:,target_col].values
        return X,y

    def train_all_ml_model(self,X,y,save_model=True,)  : 
        """
        Do time series forecasting with some model below :
        
        list_of_models =  [ensemble.RandomForestRegressor(), XGBRegressor(),svm.SVR(),
                           neighbors.KNeighborsRegressor(),linear_model.LinearRegression(),
                           linear_model.PassiveAggressiveRegressor(),linear_model.RANSACRegressor(),
                           linear_model.SGDRegressor(),linear_model.TweedieRegressor()]

        Args:
            X ([pd.Series , pd.Dataframe]): 
            y ([pd.Series , pd.Dataframe]): 
            save_model (bool, optional): if True each of the model would be saved to MODEL_OUTPUT_PATH . Defaults to True.

        Returns:
            [pd.Dataframe]: compiled result of listed models performance 
        """
        from sklearn import ensemble, linear_model,svm,neighbors
        from xgboost import XGBRegressor 

        list_of_models =  {"RandomForestRegressor":ensemble.RandomForestRegressor(),"XGBRegressor" :  XGBRegressor(),"SVRegressor" : svm.SVR(),
                           "KNeighborsRegressor":neighbors.KNeighborsRegressor(),"LinearRegression":linear_model.LinearRegression(),
                           "PassiveAggressiveRegressor":linear_model.PassiveAggressiveRegressor()
                           }

        list_of_model_result = []
        pbar = tqdm(total=100)
        for model_name in list_of_models.keys() : 
            rmse_compiled = []
            mae_compiled = []
            mape_compiled = []
            TRAIN_LENGTH = 100 
            TEST_LENGTH = 29 
            model = list_of_models.get(model_name)
            for i in range(TEST_LENGTH) : 
                
                X_TRAIN,X_VALID = X[:TRAIN_LENGTH], X[TRAIN_LENGTH:]
                Y_TRAIN,Y_VALID =  y[:TRAIN_LENGTH], y[TRAIN_LENGTH:]
                model.fit(X_TRAIN,Y_TRAIN)
                y_hat = model.predict(X_VALID)
                
                rmse_,mae_,mape_ = get_metrics(ytrue=Y_VALID,yhat=y_hat,json_output=False)
                # append all the result 
                rmse_compiled.append(rmse_)
                mae_compiled.append(mae_)
                mape_compiled.append(mape_)
                TRAIN_LENGTH +=1
                TEST_LENGTH -= 1 
    
            model_name_format = f'{model_name}'
            model_performance = {
                'RMSE Score' : np.mean(rmse_compiled) , 
                'MAE Score' : np.mean(mae_compiled) , 
                'MAPE Score' : np.mean(mape_compiled)
            }
            model_performance['model_name'] = model_name_format
            model_performance_df = pd.DataFrame(model_performance,index=[0])
            list_of_model_result.append(model_performance_df)
            if save_model : 
                model.fit(X,y)
                with save_model_path() : 
                    try : 
                        pickle.dump(model,f'{model_name_format}.pkl')
                    except TypeError : 
                        joblib.dump(model,f'{model_name_format}.pkl')
            pbar.update(math.ceil(100/len(list_of_models)))
        compiled_result_model = pd.concat(list_of_model_result,axis=0)
        return compiled_result_model

    
    