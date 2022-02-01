#inside forecast_model_NN.py 
#import required packages
import pandas as pd # dataframe operations
import pickle,joblib # for saving model 
from neuralprophet import NeuralProphet  # import model 
from metrics import get_metrics # relative import from metrics.py 
from utilities import save_model_path # relative import from utilities.py


#model params
NEURAL_PROPHET_PARAMS = { 
'weekly_seasonality' : 12,
    'num_hidden_layers' : 5 , 
    'epochs' : 500
}


#instancciating the neuralprophet 


class NNModeller : 
    """
    creating NNModeller instead of fitting directly because in development plans iam going to add another models
    """
    def prophet_modelling(self,data:pd.DataFrame,ts: str='Total Passenger',save_model=True) : 
        """function to wrap prophet model 

        Args:
            data (pd.DataFrame): forecast data
            ts (str): timeseries feature
            save_model (bool, optional): if True the model will be saved to .pkl file Defaults to True.

        Returns:
            [pd.DataFrame]: model performance recap 
        """
        #the columns requirement from both Prophet and NeuralProphet is same which require datetime feature = ds and forecast data as y 
        data.rename(columns={'Period':'ds',ts : 'y'},inplace=True)
        #instanciating the model 
        model = NeuralProphet(**NEURAL_PROPHET_PARAMS)
        #fitting the model
        model.fit(data,freq='MS')
        #creating inference
        prediction = model.predict(data)
        #get the metrics
        rmse_,mae_,mape_ = get_metrics(ytrue=prediction['y'].values, 
                    yhat=prediction['yhat1'],dict_output=False)

        from pprint import pprint
        metrics_combined = {
            'model_name' : 'NeuralProphet_hidden_layer3_epoch_3_weekly_seasonality12', 
            'RMSE Score' : rmse_, 
            'MAE Score' : mae_, 
            'MAPE Score' : mape_
        }
        pprint(metrics_combined)
        metrics_combined_df = pd.DataFrame(metrics_combined,index=[0])
    
        #saving the model 
        if save_model : 
            with save_model_path() : 
                try : 
                    pickle.dump(model,'NeuralProphet_hidden_layer3_epoch_3_weekly_seasonality12.pkl')
                except TypeError : 
                    joblib.dump(model,'NeuralProphet_hidden_layer3_epoch_3_weekly_seasonality12.pkl')
                    
        return metrics_combined_df



