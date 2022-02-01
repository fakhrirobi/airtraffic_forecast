#inside forecast_model_ML.py
#importing all required packages

#basic packages
import pandas as pd 
import numpy as np
import os 




from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler #for scaler 
from tqdm import tqdm # for progress monitoring 
from metrics import get_metrics # relative import from metrics.py 
from utilities import save_model_path # relative import from utilities.py
import joblib, pickle # for saving model by pickling / joblib
from math import ceil #



    
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
        """ function to prepare feature such as lag,month, year ( further for cyclical feature )

        Args:
            data ([pd.Dataframe,pd.Series]): data for timeseries forecasting
            ts ([str]): name of ts feature to be forecasted
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

    def create_cyclical_feature(self,data:pd.DataFrame, col_name: str, period:int, start_num:int=0)->pd.DataFrame:
        """ function to create cyclical feature 

        Args:
            data ([pd.Dataframe,pd.Series]): data for timeseries forecasting
            col_name ([str]): col name (month/year)
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
        """ wrapping all the data preparation step 

        Args:
            data ([type]): data for timeseries forecasting
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
        #splitting the dataset into X, y due to supervised approach 
        X,y  = self.train_test_split(data=final_data,target_col='Total Passenger')
        
        
        return X,y  
    
    def train_test_split(self,data,target_col,split=False) : 
        """ split the dataset into X, y 

        Args:
            data ([type]): data for timeseries forecasting
            target_col ([type]): target name (y )
            split (bool, optional): Defaults to False.

        Returns:
            [np.array]: X, y (feature and target )
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
            X ([pd.Series , pd.Dataframe]): feature
            y ([pd.Series , pd.Dataframe]): target 
            save_model (bool, optional): if True each of the model would be saved to MODEL_OUTPUT_PATH . Defaults to True.

        Returns:
            [pd.Dataframe]: compiled result of listed models performance 
        """
        #import all model candidate 
        from sklearn import ensemble, linear_model,svm,neighbors
        from xgboost import XGBRegressor 
        #creating collection of model
        list_of_models =  {"RandomForestRegressor":ensemble.RandomForestRegressor(),"XGBRegressor" :  XGBRegressor(),"SVRegressor" : svm.SVR(),
                           "KNeighborsRegressor":neighbors.KNeighborsRegressor(),"LinearRegression":linear_model.LinearRegression(),
                           "PassiveAggressiveRegressor":linear_model.PassiveAggressiveRegressor()
                           }
        #empty list as container
        list_of_model_result = []
        #initializing the progress bar
        pbar = tqdm(total=100)
        #looping over model candidate 
        for model_name in list_of_models.keys() : 
            #creating empty list of metrics for each model 
            rmse_compiled = []
            mae_compiled = []
            mape_compiled = []
            #setting the TRAIN and TEST LENGTH for crossvalidation 
            TRAIN_LENGTH = 100 
            TEST_LENGTH = 29 
            #accessing the dict of model to get str of model name 
            model = list_of_models.get(model_name)
            for i in range(TEST_LENGTH) : 
                #creating train and validation set 
                X_TRAIN,X_VALID = X[:TRAIN_LENGTH], X[TRAIN_LENGTH:]
                Y_TRAIN,Y_VALID =  y[:TRAIN_LENGTH], y[TRAIN_LENGTH:]
                #fitting model 
                model.fit(X_TRAIN,Y_TRAIN)
                #making inference 
                y_hat = model.predict(X_VALID)
                #get metrics
                rmse_,mae_,mape_ = get_metrics(ytrue=Y_VALID,yhat=y_hat,dict_output=False)
                # append all the result 
                rmse_compiled.append(rmse_)
                mae_compiled.append(mae_)
                mape_compiled.append(mape_)
                #increment train length and test length 
                TRAIN_LENGTH +=1
                TEST_LENGTH -= 1 
            #model name 
            model_name_format = f'{model_name}'
            #creating model performance dictionary 
            model_performance = {
                'RMSE Score' : np.mean(rmse_compiled) , 
                'MAE Score' : np.mean(mae_compiled) , 
                'MAPE Score' : np.mean(mape_compiled)
            }
            #adding model name 
            model_performance['model_name'] = model_name_format
            #creating dataframe of model performance 
            model_performance_df = pd.DataFrame(model_performance,index=[0])
            #appending each df to list for concat 
            list_of_model_result.append(model_performance_df)
            #fitting the model if we want to savee the model 
            if save_model : 
                model.fit(X,y)
                #change the path for saving the model 
                with save_model_path() : 
                    try : 
                        pickle.dump(model,f'{model_name_format}.pkl')
                    except TypeError : 
                        joblib.dump(model,f'{model_name_format}.pkl')
            #update progress for each loop / model 
            pbar.update(ceil(100/len(list_of_models)))
        #compiling all model result as dataframe 
        compiled_result_model = pd.concat(list_of_model_result,axis=0)
        
        return compiled_result_model

    
    