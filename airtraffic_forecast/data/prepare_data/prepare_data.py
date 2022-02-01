#inside prepare_data.py
#import required packages
from typing import Type
import pandas as pd 
import os 


class DataWrangler : 
    """
    custom class for datawrangling the original passenger airline dataset 
    functionality : 
    1. Loading the original  dataset 
    2. Preparing EDA data 
    3. Preparing forecast data
    """
    
    def load_data(self) : 
        #reading as pandas dataframe
        filepath = 'Air_Traffic_Passenger_Statistics.csv'
        data = pd.read_csv(filepath)

        return data 
    
    def prepare_eda_data(self,dataframe : pd.DataFrame) : 
        """
        Class method for preparing EDA data, wrangling steps : converting str to pd.datetime, dropping duplicates, 
        changing value for GEO Region 
        Args:
            dataframe (pd.DataFrame): Air_Traffic_Passenger_Statistics.csv path : data/original_data/Air_Traffic_Passenger_Statistics.csv

        Returns:
            data[pd.Dataframe]: Return Ready for EDA dataframe
        """
        #deep copying the dataframe to avoid changes in original data
        data = dataframe.copy(deep=True)
        #replacing airline value
        data = data.replace('United Airlines - Pre 07/01/2013', 'United Airlines')
        #changing period format from string to datetime
        data['Period'] = data['Activity Period'].astype('string')
        data['Period'] = pd.to_datetime(data['Period'], format='%Y%m')
        #dropping duplicates
        data = data.drop_duplicates(keep='first')
        #dropping Activity Period columns
        data = data.drop(columns=['Activity Period'])
        #replacing value of each GEO Region values for aggregation purpose 
        data['GEO Region'] = data['GEO Region'].replace('Canada', 'North America')
        data['GEO Region'] = data['GEO Region'].replace('US', 'North America')
        data['GEO Region'] = data['GEO Region'].replace('Australia / Oceania', 'Australia')
        data['GEO Region'] = data['GEO Region'].replace('Middle East', 'Asia')
        data['GEO Region'] = data['GEO Region'].replace('Central America', 'South America')
        data['GEO Region'] = data['GEO Region'].replace('Mexico', 'South America')
        return data
    
    def prepare_forecast_data(self,dataframe) : 
        """

        Args:
            dataframe (pd.DataFrame): Air_Traffic_Passenger_Statistics.csv path : data/original_data/Air_Traffic_Passenger_Statistics.csv

        Returns:
            data[pd.Dataframe]: Return Ready for EDA dataframe
        """
        #deep copying the dataframe to avoid changes in original data
        data = dataframe.copy(deep=True)
        #since the passenger count is not aggregated yet we need to compile it using pd.droupby
        data = data.groupby(['Period']).agg(**{'Total Passenger': ('Passenger Count', 'sum')}).reset_index()
        return data
                
                
