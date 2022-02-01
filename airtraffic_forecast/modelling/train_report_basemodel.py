from fpdf import FPDF
from numpy import size
from utilities import load_model_performance_path , save_figure_path
from tqdm import tqdm
import pandas as pd 
import plotly.express as px 
import plotly


# 




class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Helvetica', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'Title', 1, 0, 'C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Helvetica', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
        
    def export_figure(self,figure : plotly.graph_objs.Figure , output_name=None )-> None : 
         if isinstance(figure,plotly.graph_objs.Figure) : 
             #begin the process 
            figure.write_image(f'{output_name}.png')
    def create_figure(self,data : pd.DataFrame,method : str, metrics : str='MAPE'  ) -> plotly.graph_objs.Figure : 
        fig = px.bar(data, color='model_name', x='model_name', y=f'{metrics} Score', text=f'{metrics} Score', title=f'{method} model {metrics} comparison', template='ggplot2')
        fig.update_xaxes(title_text='Model Name')
        fig.update_yaxes(title_text=metrics)
        return fig
    
    
def generate_report() : 
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Times', '', 12)
    with load_model_performance_path() : 
        statistical_result = pd.read_csv('sarimax_result_.csv')
        ml_result = pd.read_csv('ml_model_result.csv')
        nn_result = pd.read_csv('nn_result.csv')
        
    #first page 
    statistical_figure_MAPE = pdf.create_figure(data=statistical_result,method='Statistical',metrics='MAPE')
    ml_figure_MAPE = pdf.create_figure(data=ml_result,method='Machine Learning',metrics='MAPE')
    nn_figure_MAPE = pdf.create_figure(data=nn_result,method='Neural Networks',metrics='MAPE')
    
    
    #second page
    
    statistical_figure_RMSE= pdf.create_figure(data=statistical_result,method='Statistical',metrics='RMSE')
    ml_figure_RMSE = pdf.create_figure(data=ml_result,method='Machine Learning',metrics='RMSE')
    nn_figure_RMSE = pdf.create_figure(data=nn_result,method='Neural Networks',metrics='RMSE')
    
    #third page
    
    statistical_figure_MAE= pdf.create_figure(data=statistical_result,method='Statistical',metrics='MAE')
    ml_figure_MAE = pdf.create_figure(data=ml_result,method='Machine Learning',metrics='MAE')
    nn_figure_MAE = pdf.create_figure(data=nn_result,method='Neural Networks',metrics='MAE')
    
    
    
    
    with save_figure_path() : 
        pdf.export_figure(figure=statistical_figure_MAPE,output_name="statistical_figure_MAPE")
        pdf.export_figure(figure=ml_figure_MAPE,output_name="ml_figure_MAPE")
        pdf.export_figure(figure=nn_figure_MAPE,output_name="nn_figure_MAPE")
        pdf.image('statistical_figure_MAPE.png',w=100,h=100)
        pdf.image('ml_figure_MAPE.png',w=100,h=100)
        pdf.image('nn_figure_MAPE.png',w=100,h=100)
        
    pdf.add_page()
    with save_figure_path() :    
        pdf.export_figure(figure=statistical_figure_RMSE,output_name="statistical_figure_RMSE")
        pdf.export_figure(figure=ml_figure_RMSE,output_name="ml_figure_RMSE")
        pdf.export_figure(figure=nn_figure_RMSE,output_name="nn_figure_RMSE")
        pdf.image('statistical_figure_RMSE.png',w=100,h=100)
        pdf.image('ml_figure_RMSE.png',w=100,h=100)
        pdf.image('nn_figure_RMSE.png',w=100,h=100)
        
    pdf.add_page()   
    with save_figure_path() :    
        pdf.export_figure(figure=statistical_figure_MAE,output_name="statistical_figure_MAE")
        pdf.export_figure(figure=ml_figure_MAE,output_name="ml_figure_MAE")
        pdf.export_figure(figure=nn_figure_MAE,output_name="nn_figure_MAE")
        pdf.image('statistical_figure_MAE.png',w=100,h=100)
        pdf.image('ml_figure_MAE.png',w=100,h=100)
        pdf.image('nn_figure_MAE.png',w=100,h=100)
    
        

    with save_figure_path() : 
        pdf.output('base_model_training.pdf', 'F')

if __name__ == '__main__' : 
    generate_report()