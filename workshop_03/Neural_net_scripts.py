# Databricks notebook source
import pandas as pd
import ipywidgets as wd
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
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from datetime import datetime as dtime
print("Imports done")

target_var = "Mobile_Traffic"


# Read Data
def fetch_data():
  df = pd.read_csv('../data/transformed_data_raw.csv', index_col=0)
  df.columns = df.columns.str.replace(' ', '_')
  df.columns = df.columns.str.replace('\[', '')
  df.columns = df.columns.str.replace('\]', '')

  return df

def set_up_layer_sliders():
  # define slider range and value
  slider_min = 1
  slider_max = 40
  slider_value = 10

  # set up slider layout
  layout = wd.Layout(width='auto', height='40px') #set width and height
  
  layer_list =  ['Layer 1','Layer 2','Layer 3']

  sliders = { 
      i : wd.IntSlider(
        min=slider_min,
        max=slider_max,
        step=1,
        description=i + ': ',
        value=slider_value,
        #layout=wd.Layout(width='60%'),
        #style={'description_width': '100%'}
      ) for i in layer_list
    }
  return sliders


def initiate_and_run_ann(X, y, hidden_layer_sizes=(2,1,3), train_size=0.75, max_iter=500):
  scores = []
  train_scores = []
  for seed in range(5):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=seed)
      
      model = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes, 
                          activation ='relu', alpha=0.0001,
                          solver='adam', learning_rate='constant',
                          max_iter= 800, n_iter_no_change =50, random_state=seed
              )
      model.fit(X_train, y_train)
      scores.append(model.score(X_test, y_test))
      train_scores.append(model.score(X_train, y_train))
      #print("Seed: {}, Score: {}".format(seed, mlr.score(X_test, y_test)))
  
  actual_train = y_train
  predicted_train = model.predict(X_train)

  actual_test = y_test
  predicted_test = model.predict(X_test)
  #print(np.average(scores))
  return scores, train_scores, actual_train, predicted_train, actual_test, predicted_test

def result_text(mae, mape):
    s = f"Mean Absolute Error: {mae} <br> " 
    s += f"On average your model predicts the mobile traffic to be {mae} GB off from the actual value<br>"
    s += f"That corresponds to {mape} % off the actual value on average <br><br>"
    if mae == 0:
        s += "Press the button for a first run"
    elif mae < 2.3:
        s += "Good job! Can you beat your own record?"
    elif mae >= 2.3:
        s += "Try again! You can do better"
    return s

def plot_pred(ax, y_predicted, y_actual):
    ax.clear()
    ax.scatter(y_predicted,y_actual)
    ax.plot([0, max(max(y_actual), max(y_predicted))], [0,max(max(y_actual), max(y_predicted))], 'red')
    ax.set_title("Mobile Traffic - ML Model")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return ax

# ================= START WIDGET =====================
plt.ioff()
fig1, ax1 = plt.subplots()
fig1.canvas.header_visible = False

sliders = set_up_layer_sliders()
slider_box = wd.HBox([
    wd.VBox([wd.HTML(v) for v in sliders]),
    wd.VBox([v for v in sliders.values()])
])

run_button = wd.Button(description="=>", layout=wd.Layout(height='50px', width='50px'))

result_widget = wd.HTML(
    value=result_text(10, 10)
)

def click_button(b):
    # do prediction, get train and test results
    X = df_raw.drop(target_var, axis=1).values
    y = df_raw[target_var].values
    ann_architecture = []
    layer_list =  ['Layer 1','Layer 2','Layer 3']
    for var_name, slider in sliders.items(): 
      layer_size = slider.value
      ann_architecture.append(layer_size)
    
    ann_architecture_tup = tuple(ann_architecture)
    print('Training your neural net')
    scores, train_scores, actual_train, predicted_train, actual_test, predicted_test = initiate_and_run_ann(X, y, hidden_layer_sizes=ann_architecture_tup)

    print('Ready!')
    plot_pred(ax1, y_predicted = predicted_test, y_actual=actual_test)
    fig1.canvas.draw()
    fig1.canvas.flush_events()

    mape = round(np.mean(np.abs((actual_test - predicted_test)/actual_test))*100,2)
    mae = round(sum(abs(actual_test- predicted_test))/len(actual_test),2)
    
    prediction_log.append((dtime.now(), ann_architecture, mape, mae))

    result_widget.value = result_text(mae, mape)

run_button.on_click(click_button)


main = wd.HBox([
    wd.VBox([slider_box, result_widget]),
    run_button,
    wd.VBox([
        fig1.canvas,
    ]),
])
    







#############################################################################################3
def run_get_slider_values():
  df_raw = fetch_data()
  target_var = "Mobile_Traffic"
  
  X = df_raw.drop(target_var, axis=1).values
  y = df_raw[target_var].values
  ann_architecture = []
  layer_list =  ['Layer 1','Layer 2','Layer 3']
  for i in layer_list: 
    layer_size = globals()[f'slider_{i}'].value
    ann_architecture.append(layer_size)
  
  ann_architecture_tup = tuple(ann_architecture)
  score = initiate_and_run_ann(X, y, hidden_layer_sizes=ann_architecture_tup)
  print(f'Yay!{score}')


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as wd
from scipy import stats
import statsmodels.formula.api as smf
from datetime import datetime as dtime
print("Imports done")
# Read Data
df_raw = pd.read_csv('../data/transformed_data_raw.csv', index_col = 0)
df_view = pd.read_csv('../data/transformed_data_for_viewing.csv', index_col = 0)

# Hide last 200 observations for testing 
df_unseen = df_raw[-200:]
df = df_raw[:-200]

variables = list(df.columns)
target_var = "Mobile Traffic"
variables.remove(target_var)

# standardizing dataframe so coefficients are -1 and 1
df_z = df.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)

# Store mean and std to transform back 
mean_std={}
for var in df.columns:
    mean_std[var]=(df[var].mean(), df[var].std())

def reverse_zscore(pandas_series, mean, std):
    '''Mean and standard deviation should be of original variable before standardization'''
    yis=pandas_series*std+mean
    return yis

original_mean, original_std = mean_std[target_var]
original_var_series = reverse_zscore(df_z[target_var], original_mean, original_std)

Y_actual = df[target_var]
print("Prepared data")

prediction_log = []

def make_sliders_A():
  # define slider range and value
  slider_min = 1
  slider_max = 10
  slider_value = 0

  # set up slider layout
  layout = wd.Layout(width='auto', height='40px') #set width and height
  
  layer_list =  ['A. Number of layers', 'A. Number of nodes']
  sliders = { 
      i : wd.IntSlider(
        min=slider_min,
        max=slider_max,
        step=1,
        #description=i + ': ',
        value=slider_value,
        #layout=wd.Layout(width='60%'),
        #style={'description_width': '100%'}
      ) for i in layer_list
    }
  
  return sliders

sliders = make_sliders_A()

def read_sliders():
    weights = {name : slider.value/10 for var_name, slider in sliders.items()}
    return weights

  



def run_ann_1():
  ann_architecture = []
  layer_list =  ['Layer 1','Layer 2','Layer 3', 'Layer 4', 'Layer 5']
  for i in layer_list: 
    layer_size = globals()[f'slider_{i}'].value
    ann_architecture.append(layer_size)
  
  ann_architecture_tup = tuple(ann_architecture)
  score = initiate_and_run_ann(hidden_layer_sizes=ann_architecture_tup)
  print(f'Yay!{score}')

def plot_manual_pred(ax, Y_pred_trans):
    ax.clear()
    ax.scatter(Y_pred_trans,Y_actual)
    ax.plot([0, max(max(Y_actual), max(Y_pred_trans))], [0,max(max(Y_actual), max(Y_pred_trans))], 'red')
    ax.set_title("Mobile Traffic - Manual Model")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return ax

def result_text(mae, mape):
    s = f"Mean Absolute Error: {mae} <br> " 
    s += f"On average your model predicts the mobile traffic to be {mae} GB off from the actual value<br>"
    s += f"That corresponds to {mape} % off the actual value on average <br><br>"
    if mae == 0:
        s += "Press the button for a first run"
    elif mae < 2.3:
        s += "Good job! Can you beat your own record?"
    elif mae >= 2.3:
        s += "Try again! You can do better"
    return s

