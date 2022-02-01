




import pandas as pd
import pickle,joblib
from neuralprophet import NeuralProphet 
from metrics import get_metrics 
from utilities import save_model_path

NEURAL_PROPHET_PARAMS = { 
'weekly_seasonality' : 12,
    'num_hidden_layers' : 3, 
    'epochs' : 1000
}


#instancciating the neuralprophet 


class NNModeller : 
    def prophet_modelling(self,data,ts,save_model=True) : 
        data.rename(columns={'Period':'ds',ts : 'y'},inplace=True)
        model = NeuralProphet(**NEURAL_PROPHET_PARAMS)
        model.fit(data,freq='MS')
        prediction = model.predict(data)
        rmse_,mae_,mape_ = get_metrics(ytrue=prediction['y'].values, 
                    yhat=prediction['yhat1'],json_output=False)

        from pprint import pprint
        metrics_combined = {
            'model_name' : 'NeuralProphet', 
            'RMSE Score' : rmse_, 
            'MAE Score' : mae_, 
            'MAPE Score' : mape_
        }
        pprint(metrics_combined)
        metrics_combined_df = pd.DataFrame(metrics_combined,index=[0])
        
        if save_model : 
            with save_model_path() : 
                try : 
                    pickle.dump(model,'NeuralProphet_model.pkl')
                except TypeError : 
                    joblib.dump(model,'NeuralProphet_model.pkl')
                    
        return metrics_combined_df
    def LSTM_modelling(self,data,ts) : 
        pass


