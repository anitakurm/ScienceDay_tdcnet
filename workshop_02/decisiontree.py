import pandas as pd
import ipywidgets as wd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from datetime import datetime as dtime
from scipy import interpolate

warnings.filterwarnings('ignore')

print("Imports done")

target_var = "Mobile_Traffic"

# Read Data Function
def fetch_data():
    df = pd.read_csv('../data/transformed_data_raw.csv', index_col=0)
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('\[', '')
    df.columns = df.columns.str.replace('\]', '')
    print(df.columns)
    return df

df_raw = fetch_data()

# Set up decision tree options using a drop-down
def set_up_decision_tree_dropdown():
    layout = wd.Layout(width='auto', height='40px')  # Set width and height

    dropdown_max_depth = wd.Dropdown(
        options=[1, 2, 3, 4, 5, 10, 20],
        value=1,
        description='Max Depth:',
        style=dict(description_width="130px"),
        layout=layout
    )

    widgets = {
        'Max Depth': dropdown_max_depth,
    }

    return widgets

def initiate_and_run_model(X, y, train_size=0.8, max_depth=None):
    seed = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=seed)

    model = DecisionTreeRegressor(random_state=seed, max_depth=max_depth)
    
    model.fit(X_train, y_train)
    
    actual_train = y_train
    predicted_train = model.predict(X_train)

    actual_test = y_test
    predicted_test = model.predict(X_test)

    return actual_train, predicted_train, actual_test, predicted_test, model

# Result Text
def train_text(train_state):
    if train_state:
        return '<div style="font-size:1.5em;font-weight:bold;display: flex;justify-content: center;text-align: center;padding-top: 0.5em; padding-bottom:0.5em">Your model is training 🏋️🦾</div>'

def result_text(mae, mape, max_depth):
    s = (
        f'<div style="font-size: 1.5em; padding-top: 1em">Mean Absolute Error:</div>'
        f'<div style="font-size: 2em; font-weight: bold;display: flex;justify-content: center;padding: 1em;"> {mae:.4f} </div>'
    )

    if mae > 0:
        s += (
            f'<div style="font-size: 1.2em">On average your model predicts the mobile traffic to be {mae:.2f} GB off from the actual value<br>'
            f'That corresponds to <span style="font-weight: bold">{mape:.2f} %</span> off the actual value on average</div>'
            f'<div style="font-size: 1.2em"> Max Depth: <span style="font-weight: bold">{max_depth}</span> </div>'
        )
        record = prediction_log_df["Mean Absolute Error"].min()
        s3 = f'<div style="font-size: 1.2em"> Your current record: <span style="font-weight: bold">{record:.4f} </span></div>'
    else:
        s3 = ''

    if mae <= 0:
        s2_val = "Press the ➡ button for a first run!"
    elif mae < 0.:
        s2_val = "Awesome job! 💪 Can you beat your own record?"
    elif mae < 1.1:
        s2_val = "You're on track, but you can do better! 😉 <br> Try again!"
    else:
        s2_val = "This is quite off 🤔 <br>Maybe try something else by resetting."

    s2 = f'<div style="{SLIDER_DESC_STYLE}">{s2_val}</div>'

    return s + s2 + s3

def plot_pred(ax, y_predicted, y_actual, title):
    ax.clear()
    ax.scatter(y_predicted, y_actual, alpha=0.6)
    limits = [min(min(y_predicted), min(y_actual)), max(max(y_predicted), max(y_actual))]
    ax.plot(limits, limits, 'red', linestyle="dashed", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return ax

def plot_prediction_log(ax):
    ax.clear()
    date_np = prediction_log_df["Time"].values
    value_np = prediction_log_df["Mean Absolute Error"].values
    date_num = mpl.dates.date2num(date_np)
    # smooth
    if len(date_np) > 1:
        date_num_smooth = np.linspace(date_num.min(), date_num.max(), 1000) 

        value_np_smooth = interpolate.pchip_interpolate(date_num, value_np, date_num_smooth)
        ax.plot(mpl.dates.num2date(date_num_smooth), value_np_smooth)
        ax.scatter(date_np, value_np, marker=".", )
    ax.set_title("Your progress")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xlabel("Time")
    return ax

# Start Widget
plt.ioff()
mpl.rcParams["font.family"] = ["Ubuntu", "sans-serif"]
mpl.rcParams["font.size"] = 10
mpl.rcParams["figure.figsize"] = (6.0, 4.0)

fig1, ax1 = plt.subplots(layout='constrained')
fig1.canvas.header_visible = False

fig2, ax2 = plt.subplots(layout='constrained')
fig2.canvas.header_visible = False

fig3, ax3 = plt.subplots(layout='constrained')
fig3.canvas.header_visible = False

fig4, ax4 = plt.subplots(figsize=(12, 8))
fig4.canvas.header_visible = False

widgets = set_up_decision_tree_dropdown()

SLIDER_DESC_STYLE = "font-size: 1.0em; font-weight:bold"

def widget_description(v):
    return f'<div style="{SLIDER_DESC_STYLE}">{v}</div>'

widget_box = wd.HBox([
    wd.VBox([wd.HTML(widget_description(v)) for v in widgets]),
    wd.VBox([v for v in widgets.values()])
])

button_style = dict(
        button_color = "royalblue", 
        font_weight="bold", 
        font_size="1.2em"
    )

button_layout = wd.Layout(height='50px', width='75px')
waiting_layout = wd.Layout(height='0px', width='0px')

run_button = wd.Button(
    icon="arrow-right",
    layout=button_layout,
    style=button_style
)

result_widget = wd.HTML(
    value=result_text(0, 0, 0)
)

X = df_raw.drop(target_var, axis=1).values
y = df_raw[target_var].values

prediction_log = []
prediction_log_df = pd.DataFrame(columns=[
    "Time",
    "Mean Absolute Error",
    "Mean Percentage Error",
    "Max Depth",
    "Train test split"
])

def click_button(b):
    # Collect user input for max depth
    max_depth = widgets['Max Depth'].value

    train_state = 1
    result_widget.value = train_text(train_state)
    run_button.layout = waiting_layout

    actual_train, predicted_train, actual_test, predicted_test, model = initiate_and_run_model(X, y, train_size=0.8, max_depth=max_depth)

    train_state = 0
    mape = np.mean(np.abs((actual_test - predicted_test) / (np.maximum(actual_test, 0.1)))) * 100
    mae = sum(abs(actual_test - predicted_test)) / len(actual_test)
    mae_train = sum(abs(actual_train - predicted_train)) / len(actual_train)

    plot_pred(ax1, y_predicted=predicted_test, y_actual=actual_test, title=f'Mobile Traffic: Predicted vs Actual - test score {mae:.3f}')
    fig1.canvas.draw()
    fig1.canvas.flush_events()
    
    plot_pred(ax2, y_predicted=predicted_train, y_actual=actual_train, title=f'Mobile Traffic: Predicted vs Actual - train score {mae_train:.3f}')
    fig2.canvas.draw()
    fig2.canvas.flush_events()

    prediction_log.append({
        "Time": dtime.now(),
        "Mean Absolute Error": mae,
        "Mean Percentage Error": mape,
        "Max Depth": max_depth,
        "Train test split": 0.8
    })
    global prediction_log_df
    prediction_log_df = pd.DataFrame(prediction_log)

    plot_prediction_log(ax3)
    fig3.canvas.draw()
    fig3.canvas.flush_events()

    run_button.layout = button_layout
    result_widget.value = result_text(mae, mape, max_depth)

    # Visualize the Decision Tree
    ax4.clear()
    plot_tree(model, feature_names=list(df_raw.drop(target_var, axis=1).columns), filled=True, ax=ax4)
    fig4.canvas.draw()
    fig4.canvas.flush_events()

run_button.on_click(click_button)

main = wd.VBox([
    wd.HTML('<div style="font-size: 2em; font-weight: bold;display: flex;justify-content: center;padding: 1em">Workshop 02 - Decision Tree Visualization</div>'),
    wd.HBox([
        wd.VBox([widget_box, result_widget, fig3.canvas]),
        wd.VBox([run_button]),
        wd.VBox([fig1.canvas, fig2.canvas])
    ]),
    wd.HBox([fig4.canvas])
])

main