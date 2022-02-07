from posixpath import basename, splitext
from setuptools import setup,find_packages
from glob import glob


setup(
    name ='airtraffic_forecast', 
    packages=find_packages('airtraffic_forecast'), 
    package_dir={'':'airtraffic_forecast'}, 
    py_modules=[splitext(basename(path))[0] for path in glob('airtraffic_forecast/*.py')], 
    package_data= {"":["*.txt","*.ipynb",".py"]}, 
    install_requires=[], 
    extras_require ={}, 
    
    
)