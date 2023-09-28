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
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#!pip install scikit-learn

# Read Data
def fetch_data():
  df = pd.read_csv('../transformed_data_raw.csv', index_col=0)
  df.columns = df.columns.str.replace(' ', '_')
  df.columns = df.columns.str.replace('\[', '')
  df.columns = df.columns.str.replace('\]', '')

  # standardizing dataframe so coefficients are -1 and 1
  df_z = df.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)

  return df



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
  layout = widgets.Layout(width='auto', height='40px') #set width and height
  
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

  return ann, score
  



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


# ================= START WIDGET =====================
plt.ioff()
fig1, ax1 = plt.subplots()
fig1.canvas.header_visible = False

slider_box = wd.HBox([
    wd.VBox([wd.HTML(v) for v in sliders]),
    wd.VBox([v for v in sliders.values()])
])

run_button = wd.Button(description="=>", layout=wd.Layout(height='50px', width='50px'))

result_widget = wd.HTML(
    value=result_text(10, 10)
)

def click_button(b):
    weights = read_sliders()
    score, ann = initiate_and_run_ann()
    Y_pred_trans = ann.predict(X_test)
    
    plot_manual_pred(ax1, Y_pred_trans)
    fig1.canvas.draw()
    fig1.canvas.flush_events()

    mape = round(np.mean(np.abs((Y_actual - Y_pred_trans)/Y_actual))*100,2)
    mae = round(sum(abs(Y_actual - Y_pred_trans))/len(Y_actual),2)
    
    prediction_log.append((dtime.now(), weights, Y_pred_trans, mape, mae))

    result_widget.value = result_text(mae, mape)

#run_button.on_click(click_button)
run_button.on_click(run_ann_1)


main = wd.HBox([
    wd.VBox([slider_box, result_widget]),
    run_button,
    wd.VBox([
    #    fig1.canvas,
    ]),
])
    