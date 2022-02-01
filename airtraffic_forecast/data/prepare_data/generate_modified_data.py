#inside generate_modified_data.py
#importing all required packages 
import prepare_data #relative package to python file : prepare_data.py which contain our custom datawrangling tool 
from contextlib import contextmanager
#importing tqdm for progress monitoring purpose
from tqdm import tqdm 

#creating contextmanager tool to changedir whenever reading our original dataset 
@contextmanager
def change_path(path) : 
    import os
    #saving prev cwd before switch 
    prev_cwd = os.getcwd()
    #changing to upper dir (data)
    os.chdir('..')
    #changing path 
    os.chdir(path)
    try : 
        yield
    finally : 
        #turning back to prev cwd path
        os.chdir(prev_cwd) 
        
    
#creating a function to wrap datawrangling step followed by saving into two dataset : 1. EDA Dataset 2. Forecast Dataset
def begin_process() : 
    #instanciating the DataWrangler class
    pbar = tqdm(total=100)
    wrangler = prepare_data.DataWrangler()
    #using contextmanager to change path temporary and read the .csv file
    with change_path('original_data') : 
        data = wrangler.load_data()
    #prepare eda data
    eda_data = wrangler.prepare_eda_data(data)
    #update progress 
    pbar.update(n=50)
    #save to ready_to_process_data folder
    with change_path('ready_to_process_data') : 
        eda_data.to_csv('prepare_for_eda.csv',index=False)
    print(eda_data.head(4))




    

    # the output should contain two columns : Activity Period and Passengger Total 
    forecast_data = wrangler.prepare_forecast_data(eda_data)
    print(forecast_data.head(4))
    #save to ready_to_process_data folder
    with change_path('ready_to_process_data') : 
        forecast_data.to_csv('forecast_data.csv',index=True)
    #update progress    
    pbar.update(n=50)
    print('\n Successfully created prepare_for_eda.csv and forecast_data.csv')
    
    
if __name__ == '__main__' : 
    begin_process()