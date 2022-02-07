from posixpath import basename, splitext
from setuptools import setup,find_packages
from glob import glob


setup(
    name ='airtraffic_forecast', 
    packages=find_packages('airtraffic_forecast'), 
    package_dir={'':'airtraffic_forecast'}, 
    py_modules=[splitext(basename(path))[0] for path in glob('airtraffic_forecast/*.py')], 
    package_data= {"":["*.txt","*.ipynb",".py"]}, 
    install_requires=[
        "statsmodels==0.13.1", 
        "numpy==1.19.5", 
        "plotly==5.3.1",  
        "tqdm==4.62.2", 
        "xgboost==1.1.1", 
        "neuralprophet==0.3.0", 
        "pandas==1.3.5", 
        "joblib==1.0.0", 
        "scikit_learn==1.0.2"
        ], 
    extras_require ={}, 
    
    
)