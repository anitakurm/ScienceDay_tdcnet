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

def make_sliders():
    slider_min = -10
    slider_max = 10
    slider_value = (slider_max+slider_min)/2
    
    layout = wd.Layout(width='auto', height='40px') #set width and height
    
    
    sliders = { 
      i : wd.IntSlider(
        min=slider_min,
        max=slider_max,
        step=1,
        #description=i.replace("_", " ") + ': ',
        value=slider_value,
        #layout=wd.Layout(width='60%'),
        #style={'description_width': '100%'}
      ) for i in variables
    }
    return sliders

sliders = make_sliders()

def read_sliders():
    weights = {var_name : slider.value/10 for var_name, slider in sliders.items()}
    return weights

def manual_predict(weights):
    # Manually calculate y = alpha * x + beta
    X = df_z.copy()
    Y = X.pop(target_var)
    for i in X.columns:
        alpha = weights[i]
        X[i] = df_z[i] * alpha 
    Y_pred = X.sum(axis=1)
    
    Y_pred_trans = reverse_zscore(Y_pred, original_mean, original_std)

    return Y_pred_trans

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
    Y_pred_trans = manual_predict(weights)
    
    plot_manual_pred(ax1, Y_pred_trans)
    fig1.canvas.draw()
    fig1.canvas.flush_events()

    mape = round(np.mean(np.abs((Y_actual - Y_pred_trans)/Y_actual))*100,2)
    mae = round(sum(abs(Y_actual - Y_pred_trans))/len(Y_actual),2)
    
    prediction_log.append((dtime.now(), weights, Y_pred_trans, mape, mae))

    result_widget.value = result_text(mae, mape)

run_button.on_click(click_button)



main = wd.HBox([
    wd.VBox([slider_box, result_widget]),
    run_button,
    wd.VBox([
        fig1.canvas,
    ]),
])
    