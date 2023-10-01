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
import matplotlib as mpl
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
  
  layer_list = ['Layer 1','Layer 2','Layer 3', 'Layer 4']

  sliders = { 
      i : wd.IntSlider(
        min=slider_min,
        max=slider_max,
        step=1,
        description='nodes: ',
        value=slider_value,
        #layout=wd.Layout(width='60%'),
        #style={'description_width': '100%'}
      ) for i in layer_list
    }
  slider_split= wd.IntSlider( min=1,
                            max=99,
                            step=1,
                            description=':',
                            value=50,
                          )
  
  sliders['% of data for train'] = slider_split
  return sliders


def initiate_and_run_ann(X, y, hidden_layer_sizes=(2,1,3), train_size=0.75, max_iter=500):

  seed = 1
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=seed)
  
  model = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes, 
                      activation ='relu', alpha=0.0001,
                      solver='adam', learning_rate='constant',
                      max_iter= 800, n_iter_no_change =50, random_state=seed
          )
  model.fit(X_train, y_train)
  #score = model.score(X_test, y_test))
  #train_score = model.score(X_train, y_train))
  
  actual_train = y_train
  predicted_train = model.predict(X_train)

  actual_test = y_test
  predicted_test = model.predict(X_test)
  #print(np.average(scores))
  return actual_train, predicted_train, actual_test, predicted_test

# ================= RESULT WIDGET ==================
def result_text(mae, mape):
    s = (
        f'<div style="font-size: 1.5em; padding-top: 1em">Mean Absolute Error:</div>' 
        f'<div style="font-size: 2em; font-weight: bold;display: flex;justify-content: center;padding: 1em;"> {mae:.3f} </div>'
    )

    # Skip this line on initial load
    if mae > 0:
        s += (
            f'<div style="font-size: 1.2em">On average your model predicts the mobile traffic to be {mae:.2f} GB off from the actual value<br>'
            f'That corresponds to <span style="font-weight: bold">{mape:.2f} %</span> off the actual value on average</div>'
        )
        record = prediction_log_df.loc[:,"Mean Absolute Error"].min()
        s3 = f'<div style="font-size: 1.2em"> Your current record: <span style="font-weight: bold">{record:.3f} </span></div>'
    else:
        s3 = ''
    
    if mae <= 0:
        s2_val = "Press the -> button for a first run!"
    elif mae < 1.7:
        s2_val = "Awesome job! Can you beat your own record?"
    elif mae < 3.2:
        s2_val = "You're on track, but you can do better.<br> Try again!"
    else:
        s2_val = "This is quite off :( <br>Maybe try something else by resetting."

    s2 = f'<div style="font-size:1.5em;font-weight:bold;display: flex;justify-content: center;text-align: center;padding-top: 0.5em; padding-bottom:0.5em">{s2_val}</div>'
    
    return s + s2 + s3

def plot_pred(ax, y_predicted, y_actual, title):
    ax.clear()
    ax.scatter(y_predicted,y_actual, alpha=0.6)
    limits = [min(min(y), min(y)), max(max(y), max(y))]
    ax.plot(limits, limits, 'red', linestyle="dashed", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return ax

# ================= START WIDGET =====================
plt.ioff()
mpl.rcParams["font.family"] = ["Ubuntu", "sans-serif"]
mpl.rcParams["font.size"] = 10
mpl.rcParams["figure.figsize"] = (6.0, 4.0) 

fig1, ax1 = plt.subplots(layout='constrained')
fig1.canvas.header_visible = False

fig2, ax2 = plt.subplots(layout='constrained')
fig2.canvas.header_visible = False

sliders = set_up_layer_sliders()

slider_box = wd.HBox([
    wd.VBox([wd.HTML(v) for v in sliders]),
    wd.VBox([v for v in sliders.values()]),
])




button_style = dict(
        button_color = "royalblue", 
        font_weight="bold", 
        font_size="1.2em"
    )
run_button = wd.Button(
    #description="=>", 
    icon="arrow-right",
    layout=wd.Layout(height='50px', width='75px'),
    style=button_style
)

result_widget = wd.HTML(
    value=result_text(0, 0)
)
target_var = "Mobile Traffic"
df_raw = pd.read_csv('../data/transformed_data_raw.csv', index_col = 0)
X = df_raw.drop(target_var, axis=1).values
y = df_raw[target_var].values

prediction_log = []
prediction_log_df = pd.DataFrame(columns = [
    "Time",
    "Mean Absolute Error", 
    "Mean Percentage Error",
    "Architecture"
])

def click_button(b):
    # do prediction, get train and test results
    ann_architecture = []
    for var_name, slider in sliders.items(): 
      if 'Layer' in var_name:
         layer_size = slider.value
         ann_architecture.append(layer_size)
      else:
          train_size = slider.value/100
    
    ann_architecture_tup = tuple(ann_architecture)
    print(f'Training your neural net {ann_architecture_tup}')
    actual_train, predicted_train, actual_test, predicted_test = initiate_and_run_ann(X, y, train_size=train_size, hidden_layer_sizes=ann_architecture_tup)

    print('Ready!')
    plot_pred(ax1, y_predicted = predicted_test, y_actual=actual_test, title='Mobile Traffic - Neural Net (Test)')
    fig1.canvas.draw()
    fig1.canvas.flush_events()
    
    plot_pred(ax2, y_predicted = predicted_train, y_actual=actual_train, title='Mobile Traffic - Neural Net (Train)')
    fig2.canvas.draw()
    fig2.canvas.flush_events()
    mape = np.mean(np.abs((actual_test - predicted_test)/(np.maximum(actual_test, 0.1))))*100
    mae = sum(abs(actual_test- predicted_test))/len(actual_test)
    
    prediction_log.append({
        "Time": dtime.now(), 
        "Mean Absolute Error": mae, 
        "Mean Percentage Error" : mape,
        "Architecture" : ann_architecture
    })
    global prediction_log_df
    prediction_log_df = pd.DataFrame(prediction_log)

    result_widget.value = result_text(mae, mape)

run_button.on_click(click_button)


main = wd.VBox([
    wd.HTML('<div style="font-size: 2em; font-weight: bold;display: flex;justify-content: center;padding: 1em">Workshop 03 - Neural Networks</div>'),
    wd.HBox([
        wd.VBox([slider_box, result_widget]),
        wd.VBox([
           run_button,
           #reset_button
        ]),
        wd.VBox([
            fig1.canvas,
            fig2.canvas
        ]),
    ])
])
    
