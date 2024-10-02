# Required Imports
from io import BytesIO
from matplotlib import pyplot as plt
from math import cos, sin, atan
import pandas as pd
import ipywidgets as wd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy import interpolate
import statsmodels.formula.api as smf
from datetime import datetime as dtime
from PIL import Image as PILImage

print("Imports done")

# Neural Network Visualization classes
class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius, ax):
        circle = plt.Circle((self.x, self.y), radius=neuron_radius, fill=False, color='black')
        ax.add_patch(circle)

class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__init_neurons(number_of_neurons)

    def __init_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for _ in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, ax):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = plt.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), 
                          (neuron1.y - y_adjustment, neuron2.y + y_adjustment), color='black')
        ax.add_line(line)

    def draw(self, layerType, ax):
        for neuron in self.neurons:
            neuron.draw(self.neuron_radius, ax)
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, ax)
        # Write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            ax.text(x_text, self.y, 'Input Layer', fontsize=12)
        elif layerType == -1:
            ax.text(x_text, self.y, 'Output Layer', fontsize=12)
        else:
            ax.text(x_text, self.y, f'Hidden Layer {layerType}', fontsize=12)

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []

    def add_layer(self, number_of_neurons ):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def draw(self, ax):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw(i, ax)
        ax.axis('scaled')
        ax.axis('off')
        ax.set_title('Neural Network architecture', fontsize=15)

class DrawNN():
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def draw(self, ax):
        widest_layer = max(self.neural_network)
        network = NeuralNetwork(widest_layer)
        for l in self.neural_network:
            network.add_layer(l)
        network.draw(ax)

# Neural Network training functions and setup
def fetch_data():
    df = pd.read_csv('../data/transformed_data_raw.csv', index_col=0)
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('\[', '') 
    df.columns = df.columns.str.replace('\]', '')
    return df

def set_up_layer_sliders():
    slider_min = 1
    slider_max = 30
    slider_value = 5
    layout = wd.Layout(width='auto', height='40px')
    layer_list = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
    sliders = {
        i: wd.IntSlider(
            min=slider_min,
            max=slider_max,
            step=1,
            value=slider_value,
            style=dict(handle_color="royalblue"),
        ) for i in layer_list
    }
    return sliders

def initiate_and_run_ann(X, y, hidden_layer_sizes=(2,1,3), max_iter=100):
    train_size = 0.80
    seed = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=seed)
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                         activation='relu', alpha=0.0001,
                         solver='adam', learning_rate='constant',
                         max_iter=max_iter, n_iter_no_change=25, random_state=seed)
    model.fit(X_train, y_train)
    actual_train = y_train
    predicted_train = model.predict(X_train)
    actual_test = y_test
    predicted_test = model.predict(X_test)
    return actual_train, predicted_train, actual_test, predicted_test

# Result Widgets
def train_text(train_state):
    if train_state:
        s = '<div style="font-size:1.5em;font-weight:bold;display: flex;justify-content: center;text-align: center;padding-top: 0.5em; padding-bottom:0.5em">Your Neural Network model is training 🏋️🦾</div>'
    return s
    
def result_text(mae, mape):
    s = (
        f'<div style="font-size: 1.5em; padding-top: 1em">Mean Absolute Error:</div>' 
        f'<div style="font-size: 2em; font-weight: bold;display: flex;justify-content: center;padding: 1em;"> {mae:.4f} </div>'
    )
    if mae > 0:
        s += (
            f'<div style="font-size: 1.2em">On average your model predicts the mobile traffic to be {mae:.2f} GB off from the actual value<br>'
            f'That corresponds to <span style="font-weight: bold">{mape:.2f} %</span> off the actual value on average</div>'
        )
        record = prediction_log_df.loc[:,"Mean Absolute Error"].min()
        s3 = f'<div style="font-size: 1.2em"> Your current record: <span style="font-weight: bold">{record:.4f} </span></div>'
    else:
        s3 = ''
    
    if mae <= 0:
        s2_val = "Press the ➡ button for a first run!"
    elif mae < 1.04:
        s2_val = "Awesome job! 💪 Can you beat your own record?"
    elif mae < 1.1:
        s2_val = "You're on track, but you can do better! 😉 <br> Try again!"
    else:
        s2_val = "This is quite off 🤔 <br>Maybe try something else by resetting."
    s2 = f'<div style="font-size:1.5em;font-weight:bold;display: flex;justify-content: center;text-align: center;padding-top: 0.5em; padding-bottom:0.5em">{s2_val}</div>'
    return s + s2 + s3

def plot_pred(ax, y_predicted, y_actual, title):
    ax.clear()
    ax.scatter(y_predicted, y_actual, alpha=0.6)
    limits = [min(min(y), min(y)), max(max(y), max(y))]
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
    if len(date_np) > 1:
        date_num_smooth = np.linspace(date_num.min(), date_num.max(), 1000)
        value_np_smooth = interpolate.pchip_interpolate(date_num, value_np, date_num_smooth)
        ax.plot(mpl.dates.num2date(date_num_smooth), value_np_smooth)
        ax.scatter(date_np, value_np, marker=".", )
    ax.set_title("Your progress")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xlabel("Time")
    return ax

# Start Widgets
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

sliders = set_up_layer_sliders()
SLIDER_DESC_STYLE = "font-size: 1.0em; font-weight:bold"

def slider_description(v):
    return f'<div style="{SLIDER_DESC_STYLE}">{v}</div>'

slider_box = wd.HBox([
    wd.VBox([wd.HTML(slider_description(v)) for v in sliders]),
    wd.VBox([v for v in sliders.values()])
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

result_widget = wd.HTML(value=result_text(0, 0))

network_output = wd.Output()

target_var = "Mobile Traffic"
df_raw = pd.read_csv('../data/transformed_data_raw.csv', index_col=0)
X = df_raw.drop(target_var, axis=1).values
y = df_raw[target_var].values

prediction_log = []
prediction_log_df = pd.DataFrame(columns=[
    "Time",
    "Mean Absolute Error",
    "Mean Percentage Error",
    "Architecture"
])

def verify():
    record_log = prediction_log_df.sort_values("Mean Absolute Error").iloc[0, :]
    print(f"The record log:")
    print(record_log)
    actual_train, predicted_train, actual_test, predicted_test = initiate_and_run_ann(X, y,
                                                                                      hidden_layer_sizes=record_log["Architecture"],
                                                                                      max_iter=100)
    mae = sum(abs(actual_test - predicted_test)) / len(actual_test)
    print(f"The calculated MAE: {mae}")

def draw_network_image(architecture):
    with network_output:
        network_output.clear_output(wait=True)
        fig_nn, ax_nn = plt.subplots(figsize=(8, 8))
        DrawNN(architecture).draw(ax_nn)
        plt.show()

def click_button(b):
    ann_architecture = []
    for var_name, slider in sliders.items():
        if 'Layer' in var_name:
            layer_size = slider.value
            ann_architecture.append(layer_size)

    ann_architecture_tup = tuple(ann_architecture)
    train_state = 1
    result_widget.value = train_text(train_state)
    run_button.layout = waiting_layout

    actual_train, predicted_train, actual_test, predicted_test = initiate_and_run_ann(X, y,
                                                                                      hidden_layer_sizes=ann_architecture_tup,
                                                                                      max_iter=100)
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
        "Architecture": ann_architecture
    })
    global prediction_log_df
    prediction_log_df = pd.DataFrame(prediction_log)

    plot_prediction_log(ax3)
    fig3.canvas.draw()
    fig3.canvas.flush_events()

    draw_network_image(ann_architecture)
    
    run_button.layout = button_layout
    result_widget.value = result_text(mae, mape)

run_button.on_click(click_button)

main = wd.VBox([
    wd.HTML('<div style="font-size: 2em; font-weight: bold; display: flex; justify-content: center; padding: 1em">Workshop 02 - Neural Networks</div>'),
    wd.HBox([
        wd.VBox([slider_box, result_widget, fig3.canvas]),
        wd.VBox([
            run_button,
        ]),
        wd.VBox([
            fig1.canvas,
            fig2.canvas
        ]),
    ]),
    wd.VBox([
        wd.HTML('<div style="font-size: 2em; font-weight: bold; display: flex; justify-content: center; padding: 1em">Neural Network Visualization</div>'),
        network_output
    ])
])

main