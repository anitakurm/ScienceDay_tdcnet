print("Importing modules")
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
prediction_log_df = pd.DataFrame(columns = [
    "Time",
    "Mean Absolute Error", 
    "Mean Percentage Error",
    "Weights"
])
# =============== PREDICTION ====================

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

# =============== SLIDERS ====================
SLIDER_DESC_STYLE = "font-size: 1.0em; font-weight:bold"

def make_sliders():
    slider_min = -8
    slider_max = 8
    slider_value = (slider_max+slider_min)/2
    
    layout = wd.Layout(width='auto', height='40px') #set width and height
    
    sliders = { 
      i : wd.FloatSlider(
        min=slider_min,
        max=slider_max,
        step=0.5,
        #description=i.replace("_", " ") + ': ',
        value=slider_value,
        #layout=wd.Layout(width='60%'),
        style=dict(handle_color="royalblue"),
      ) for i in variables
    }
    return sliders

sliders = make_sliders()

def reset_sliders(b):
    for slider in sliders.values():
        slider.value = 0

def read_sliders():
    weights = {var_name : slider.value/10 for var_name, slider in sliders.items()}
    return weights
          
def slider_description(v):
    return f'<div style="{SLIDER_DESC_STYLE}">{v}</div>'

slider_box = wd.HBox([
    wd.VBox([wd.HTML(slider_description(v)) for v in sliders]),
    wd.VBox([v for v in sliders.values()])
])

# ==================== PLOTS ======================
print("Creating graphs")
plt.ioff()
fig1, ax1 = plt.subplots()
fig1.canvas.header_visible = False

fig2, ax2 = plt.subplots()
fig2.canvas.header_visible = False

def plot_manual_pred(ax, Y_pred_trans):
    ax.clear()
    ax.scatter(Y_pred_trans,Y_actual)
    ax.plot([0, max(max(Y_actual), max(Y_pred_trans))], [0,max(max(Y_actual), max(Y_pred_trans))], 'red')
    ax.set_title("Mobile Traffic - Manual Model")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return ax

def plot_prediction_log(ax):
    ax.clear()
    ax.plot(prediction_log_df["Time"], prediction_log_df["Mean Absolute Error"])
    ax.set_title("Your progress")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xlabel("Time")
    return ax

# ================== BUTTONS =======================
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

reset_button = wd.Button(
    description="Reset", 
    layout=wd.Layout(height='50px', width='75px'),
    style=button_style
)
reset_button.on_click(reset_sliders)

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
    elif mae < 2.2:
        s2_val = "Awesome job! Can you beat your own record?"
    elif mae < 3.5:
        s2_val = "You're on track, but you can do better.<br> Try again!"
    else:
        s2_val = "This is quite off :( <br>Maybe try something else by resetting."

    s2 = f'<div style="font-size:1.5em;font-weight:bold;display: flex;justify-content: center;text-align: center;padding-top: 0.5em; padding-bottom:0.5em">{s2_val}</div>'
    
    return s + s2 + s3

result_widget = wd.HTML(
    value=result_text(0, 0)
)

# ================ BUTTON RUN FUNCTION =================
def click_button(b):
    weights = read_sliders()
    Y_pred_trans = manual_predict(weights)
    
    plot_manual_pred(ax1, Y_pred_trans)
    fig1.canvas.draw()
    fig1.canvas.flush_events()

    mape = np.mean(np.abs((Y_actual - Y_pred_trans)/(np.maximum(Y_actual, 0.1))))*100
    mae = sum(abs(Y_actual - Y_pred_trans))/len(Y_actual)
    
    prediction_log.append({
        "Time": dtime.now(), 
        "Mean Absolute Error": mae, 
        "Mean Percentage Error" : mape,
        "Weights" : weights
    })

    global prediction_log_df
    prediction_log_df = pd.DataFrame(prediction_log)

    plot_prediction_log(ax2)
    fig2.canvas.draw()
    fig2.canvas.flush_events()

    result_widget.value = result_text(mae, mape)

run_button.on_click(click_button)

main = wd.VBox([
    wd.HTML('<div style="font-size: 2em; font-weight: bold;display: flex;justify-content: center;padding: 1em">Workshop 02 - Predict Mobile Traffic</div>'),
    wd.HBox([
        wd.VBox([slider_box, result_widget]),
        wd.VBox([run_button, reset_button]),
        wd.VBox([
            fig1.canvas,
            fig2.canvas
        ]),
    ])
])
    