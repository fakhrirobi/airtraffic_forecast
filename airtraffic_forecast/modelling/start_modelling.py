#inside start_modelling.py
#importing all packages 
import argparse # for cli argument parsing -- separating each method training : statistical , ML, NN 
from datetime import datetime
from tqdm import tqdm # progress monitoring 
from math import ceil


#importing all method 
from forecast_model_statistical import * 
from forecast_model_ML import MLForecastingModel
from forecast_model_NN import  NNModeller

#creating function to handle argparse
def initialize_argparse() : 
    parser = argparse.ArgumentParser()
    #adding argument for each method
    parser.add_argument('--method', choices=['statistical', 'ml', 'NN'], required=True)
    args = parser.parse_args()
    return args


#wrapping up all process 
def start() : 
    #show selection of model 
    method_args = initialize_argparse()

    if method_args.method == 'statistical' : 
        #import the utilities 
        from utilities import stationary_test,transform_data,data_load_path
        #reading the dataset 
        with data_load_path('ready_to_process_data') : 
            data = pd.read_csv('forecast_data.csv')
        #conducting stationary test 
        stationary_result = stationary_test(data=data,ts='Total Passenger')
        #performing some logic
        if stationary_result == False : 
            #do some data transformation for complying stationarity 
            transformed_data = transform_data(data=data,ts='Total Passenger')

            stationary_feature = ['Period']
            list_of_feature = (x for x in transformed_data.columns if x not in  stationary_feature)
            for feature in list_of_feature : 

                stationary_result = stationary_test(data=transformed_data,ts=feature)
                if stationary_result ==False  : 
                    stationary_feature.append(feature)
                else : 
                    continue 
            #creating new stationary data after re test 
            stationary_data = transformed_data.loc[:,stationary_feature]
            #instanciating the model 
            sarima_model = SarimaxModel()
            sarimax_model_result = {'model_name':None, 'RMSE Score':None , 'MAE Score':None , 'SMAPE Score':None}
            list_df = []
            list_stationary_features = [x for x in stationary_data.columns if x not in  ['Period','Total Passenger']]
            pbar = tqdm(total=100)
            #looping for each feature(after transformed)
            for ts_feature in list_stationary_features : 
                
                model_result = sarima_model.train_model(data=stationary_data,ts=ts_feature,save_model=True,custom_name=True)
                model_result.drop_duplicates(inplace=True)
                list_df.append(model_result)
                pbar.update(ceil(100/len(list_stationary_features))) #update progress 
            
            #return model output 
            sarimax_model_result_df = pd.DataFrame(sarimax_model_result,index=[1])
            sarimax_result_combined = pd.concat(list_df,axis=0)
            print('Model Performance : ')
            print(sarimax_result_combined) 
            sarimax_result_combined.to_csv(f'model_performance/sarimax_result_.csv',index=False)
            
    elif method_args.method == 'ml' :  
        from utilities import data_load_path
        print('you choose ml method')
        #instanciating the model 
        ml_modeller  = MLForecastingModel()
        #read the dataset 
        with data_load_path('ready_to_process_data') : 
            data = pd.read_csv('forecast_data.csv') 
        data['Period'] = pd.to_datetime(data['Period'])
        #do some ml prep : generating lag, cyclical feature , etc. 
        X,y = ml_modeller.start_prepare_data(data=data)
        # train all candidate model
        ml_result = ml_modeller.train_all_ml_model(X,y)
        #saving the result 
        ml_result.to_csv(f'model_performance/ml_model_result.csv',index=False)
        print(ml_result.head())
            
    elif method_args.method == 'NN' :  
        from utilities import data_load_path
        #read the dataset 
        with data_load_path('ready_to_process_data') : 
            data = pd.read_csv('forecast_data.csv') 
        print(data.columns)
        #instanciating the model 
        nn_result = NNModeller().prophet_modelling(data=data,ts='Total Passenger')
        #save the result 
        nn_result.to_csv(f'model_performance/nn_result.csv',index=False) 
    else : 
        print('We have not add another method')
        

    



if __name__ == '__main__' : 
    start()