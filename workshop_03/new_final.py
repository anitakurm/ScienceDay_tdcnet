print("Importing modules")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import ipywidgets as wd
from scipy import stats, interpolate
import statsmodels.api as sm  # Import statsmodels
from sklearn.model_selection import KFold
from datetime import datetime as dtime

print("Imports done")
# Read Data
df_raw = pd.read_csv("../data/transformed_data_raw.csv", index_col=0)
df_view = pd.read_csv("../data/transformed_data_for_viewing.csv", index_col=0)

# Hide last 200 observations for testing
df_unseen = df_raw[-200:]
df = df_raw[:-200]

variables = list(df.columns)
target_var = "Mobile Traffic"
variables.remove(target_var)

# standardizing dataframe so coefficients are -1 and 1
df_z = df.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)

# Store mean and std to transform back
mean_std = {}
for var in df.columns:
    mean_std[var] = (df[var].mean(), df[var].std())

def reverse_zscore(pandas_series, mean, std):
    """Mean and standard deviation should be of original variable before standardization"""
    yis = pandas_series * std + mean
    return yis

original_mean, original_std = mean_std[target_var]
original_var_series = reverse_zscore(df_z[target_var], original_mean, original_std)

# Global Y_actual
Y_actual = df[target_var]
print("Prepared data")

prediction_log = []
prediction_log_df = pd.DataFrame(
    columns=["Time", "Mean Absolute Error", "Mean Percentage Error", "Weights"]
)

# =============== PREDICTION ====================

def manual_predict(weights, df_z):
    # Manually calculate y = alpha * x + beta
    X = df_z.copy()
    Y = X.pop(target_var)
    for i in X.columns:
        alpha = weights[i]
        X[i] = df_z[i] * alpha
    Y_pred = X.sum(axis=1)

    Y_pred_trans = reverse_zscore(Y_pred, original_mean, original_std)

    return Y_pred_trans

def auto_fit_model(df_z_train):
    X = df_z_train.copy()
    Y = X.pop(target_var)
    X = sm.add_constant(X)  # Add constant term for intercept
    model = sm.OLS(Y, X).fit()
    return model.params.drop('const')

def perform_cross_validation(df_z, n_splits=3):
    kf = KFold(n_splits=n_splits)
    maes = []
    mapes = []
    trained_weights = []

    for train_index, val_index in kf.split(df_z):
        df_train_z, df_val_z = df_z.iloc[train_index], df_z.iloc[val_index]
        Y_val = df.iloc[val_index][target_var]

        # Fit the model on the training split
        weights = auto_fit_model(df_train_z)
        trained_weights.append(weights)

        # Predict on the validation split
        Y_pred_val_trans = manual_predict(weights, df_val_z)

        # Evaluate the predictions
        mae = calc_mae(Y_val, Y_pred_val_trans)
        mape = calc_mape(Y_val, Y_pred_val_trans)
        maes.append(mae)
        mapes.append(mape)

    # Average the weights
    avg_weights = sum(trained_weights) / n_splits

    # Return average MAE and MAPE
    return np.mean(maes), np.mean(mapes), avg_weights

def verify():
    record_log = prediction_log_df.sort_values("Mean Absolute Error").iloc[0, :]
    w = record_log.pop("Weights")
    print(f"The record log:")
    print(record_log)
    print(f"with Weights: ")
    print(pd.Series(w))
    y_p = manual_predict(w, df_z)
    mae = calc_mae(Y_actual, y_p)
    print(f"The calculated MAE: {mae}")

# =============== SLIDERS ====================
SLIDER_DESC_STYLE = "font-size: 1.0em; font-weight:bold"

def make_sliders_num():
    slider_min = -10
    slider_max = 10
    slider_value = (slider_max + slider_min) / 2

    sliders = {
        i: wd.FloatSlider(
            min=slider_min,
            max=slider_max,
            step=0.5,
            value=slider_value,
            style=dict(handle_color="royalblue"),
        )
        for i in variables
    }
    return sliders

sliders = make_sliders_num()

def reset_sliders(b):
    for slider in sliders.values():
        slider.value = 0

def read_sliders():
    weights = {var_name: slider.value / 10 for var_name, slider in sliders.items()}
    return weights

def slider_description(v):
    return f'<div style="{SLIDER_DESC_STYLE}">{v}</div>'

slider_box = wd.HBox(
    [
        wd.VBox([wd.HTML(slider_description(v)) for v in sliders]),
        wd.VBox([v for v in sliders.values()]),
    ]
)

# ==================== PLOTS ======================
print("Creating graphs")
plt.ioff()
mpl.rcParams["font.family"] = ["Ubuntu", "sans-serif"]
mpl.rcParams["font.size"] = 10
mpl.rcParams["figure.figsize"] = (6.0, 4.0)

fig1, ax1 = plt.subplots(layout="constrained")
fig1.canvas.header_visible = False
fig1.canvas.capture_scroll = False

fig2, ax2 = plt.subplots(layout="constrained")
fig2.canvas.header_visible = False
fig2.canvas.capture_scroll = False

def plot_manual_pred(ax, Y_pred_trans, Y_actual_subset):
    ax.clear()
    ax.scatter(Y_pred_trans, Y_actual_subset, alpha=0.6)
    limits = [
        min(min(Y_actual_subset), min(Y_pred_trans)),
        max(max(Y_actual_subset), max(Y_pred_trans)),
    ]
    ax.plot(limits, limits, "red", linestyle="dashed", linewidth=0.8)
    ax.set_title("Mobile Traffic - Manual Model")
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
        value_np_smooth = interpolate.pchip_interpolate(
            date_num, value_np, date_num_smooth
        )
        ax.plot(mpl.dates.num2date(date_num_smooth), value_np_smooth)
        ax.scatter(
            date_np,
            value_np,
            marker=".",
        )
    ax.set_title("Your progress")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xlabel("Time")
    return ax

# ================== BUTTONS =======================
button_style = dict(button_color="royalblue", font_weight="bold", font_size="1.2em")
run_button = wd.Button(
    icon="arrow-right",
    layout=wd.Layout(height="50px", width="75px"),
    style=button_style,
)

reset_button = wd.Button(
    description="Reset",
    layout=wd.Layout(height="50px", width="75px"),
    style=button_style,
)
reset_button.on_click(reset_sliders)


# ================ INTRO WIDGET ====================
intro_text = """
<div style="font-size: 1.6em; padding: 1em">
    <p>
        <strong>Hej 8. klassere!</strong> 😊
    </p>
    <p>
        Har du nogensinde tænkt på, hvordan vi kan forudsige mobiltrafik for at planlægge de bedste mobilpakker til kunderne og se, om vi skal opgradere tårne i området?
    </p>
    <p>
        Det er her <strong>Machine Learning</strong> kommer ind! 🌟
    </p>
    <p>
        Machine Learning gør det muligt for computere at lære af eksempler. I denne øvelse vil vi vise, hvordan computere kan beregne de bedste vægte for at forudsige mobiltrafik, uden at vi skal gætte os frem.
    </p>
    <p>
        Jo flere eksempler vi giver computeren, jo bedre bliver dens forudsigelser! 🎯
    </p>
    <p>
        Lad os komme i gang og se, hvor kraftfuld Machine Learning er. Klar til at lade maskiner lære? Lad os gå i gang! 🚀
    </p>
</div>
"""

intro_widget = wd.HTML(value=intro_text)

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
        record = prediction_log_df.loc[:, "Mean Absolute Error"].min()
        s3 = f'<div style="font-size: 1.2em"> Your current record: <span style="font-weight: bold">{record:.3f} </span></div>'
    else:
        s3 = ""

    if mae <= 0:
        s2_val = "Press the ➡ button for a first run!"
    elif mae < 1:
        s2_val = "WOW 😱 You should become a data scientist!"
    elif mae < 1.5:
        s2_val = "Good job! 💪 Can you beat your own record?"
    elif mae < 3.2:
        s2_val = "You're on track, but you can do better! 😉 <br> Try again!"
    else:
        s2_val = "This is quite off 🤔 <br>Maybe try something else by resetting."

    s2 = f'<div style="font-size:1.5em;font-weight:bold;display: flex;justify-content: center;text-align: center;padding-top: 0.5em; padding-bottom:0.5em">{s2_val}</div>'

    return s + s2 + s3

result_widget = wd.HTML(value=result_text(0, 0))

def calc_mape(Y_a, Y_p):
    return np.mean(np.abs((Y_a - Y_p) / (np.maximum(Y_a, 0.1)))) * 100

def calc_mae(Y_a, Y_p):
    return sum(abs(Y_a - Y_p)) / len(Y_a)

def update_sliders_to_optimal(weights):
    for var_name, slider in sliders.items():
        slider.value = weights[var_name] * 10  # Undo the earlier division by 10

def click_button(b):
    # Get the selected percentage of training data
    training_percentage = float(training_data_selector.value.strip('%')) / 100.0
    n_train = int(len(df_z) * training_percentage)
    df_train_z = df_z.iloc[:n_train]

    avg_mae, avg_mape, optimal_weights = perform_cross_validation(df_train_z)
    update_sliders_to_optimal(optimal_weights)

    # Now predict with these optimal weights using the training data
    Y_train_actual = df.iloc[:n_train][target_var]
    Y_pred_trans = manual_predict(optimal_weights, df_train_z)

    plot_manual_pred(ax1, Y_pred_trans, Y_train_actual)
    fig1.canvas.draw()
    fig1.canvas.flush_events()

    prediction_log.append(
        {
            "Time": dtime.now(),
            "Mean Absolute Error": avg_mae,
            "Mean Percentage Error": avg_mape,
            "Weights": optimal_weights.to_dict(),  # Convert Series to dictionary
        }
    )

    global prediction_log_df
    prediction_log_df = pd.DataFrame(prediction_log)

    plot_prediction_log(ax2)
    fig2.canvas.draw()
    fig2.canvas.flush_events()

    result_widget.value = result_text(avg_mae, avg_mape)

run_button.on_click(click_button)

# Dropdown for selecting the amount of training data
training_data_selector = wd.Dropdown(
    options=["30%", "50%", "80%"],
    value="30%",
    description="Training Data:",
    style=dict(description_width='initial'),
)

# Main layout
main = wd.VBox(
    [   
        wd.HTML(
            '<div style="font-size: 2em; font-weight: bold;display: flex;justify-content: center;padding: 1em">Workshop 03 - Let Machines Learn </div>'
        ),
        intro_widget,
        training_data_selector,
        wd.HBox(
            [
                wd.VBox([slider_box, result_widget]),
                wd.VBox([run_button, reset_button]),
                wd.VBox([fig1.canvas, fig2.canvas]),
            ]
        ),
    ]
)

# Display the main layout
main