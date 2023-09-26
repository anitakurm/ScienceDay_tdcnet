# Databricks notebook source
import pandas as pd
import ipywidgets as widgets
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer


# Read Data
def fetch_data():
  df = pd.read_csv('../transformed_data_raw.csv', index_col=0)
  df.columns = df.columns.str.replace(' ', '_')
  df.columns = df.columns.str.replace('\[', '')
  df.columns = df.columns.str.replace('\]', '')

  # standardizing dataframe so coefficients are -1 and 1
  df_z = df.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)

  return df

def initiate_and_run_ann(hidden_layer_sizes=(2,1,3), alpha=0.0001, train_size=0.75, max_iter=500):

  # Read Data
  df = fetch_data()
  X = df.drop('Mobile_Traffic', axis=1)
  y = df['Mobile_Traffic']

  # split train and test data
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
  
  # initiate ANN
  ann = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                     alpha=alpha, 
                     activation='relu', 
                     solver='lbfgs',    
                     max_iter=max_iter,
                     random_state=42).fit(X_train, y_train)
  
  # get AUC score instead!!
  score = ann.score(X_test, y_test)

  return score

def set_up_layer_sliders():
  # define slider range and value
  slider_min = 1
  slider_max = 10
  slider_value = 0

  # set up slider layout
  layout = widgets.Layout(width='auto', height='40px') #set width and height
  
  layer_list =  ['Layer 1','Layer 2','Layer 3', 'Layer 4', 'Layer 5']
  for i in layer_list:
    globals()[f'slider_{i}'] = widgets.IntSlider(
                                    min=slider_min,
                                    max=slider_max,
                                    step=1,
                                    description=i + ': ',
                                    value=slider_value,
                                    layout=widgets.Layout(width='40%'),
                                    style= {'description_width': '40%'}
                                     )
  for i in layer_list: 
    display(globals()[f'slider_{i}'])

  



def run_get_slider_values():
  ann_architecture = []
  layer_list =  ['Layer 1','Layer 2','Layer 3', 'Layer 4', 'Layer 5']
  for i in layer_list: 
    layer_size = globals()[f'slider_{i}'].value
    ann_architecture.append(layer_size)
  
  ann_architecture_tup = tuple(ann_architecture)
  score = initiate_and_run_ann(hidden_layer_sizes=ann_architecture_tup)
  print(f'Yay!{score}')