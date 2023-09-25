import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as wd

plt.ioff()

features = ['Number Of Residents', 'Average Age', 'Distance To Nearest Tower [m]', 'Number Of Phones']
feature_sliders = {
    v: wd.FloatSlider(min=0.02, max=2.0) for v in features
}


feature_sliders.values()


slider_box = wd.HBox([
    wd.VBox([wd.HTML(v) for v in feature_sliders]),
    wd.VBox([v for v in feature_sliders.values()])
])

format_count = lambda x: f"Try no.: <b>{x}</b>!"
counter_widget = wd.HTML(
    value=format_count(0),
)

fig, ax = plt.subplots()
fig.canvas.header_visible = False
#fig.canvas.layout.max_width = '200px'
x = np.linspace(0, 20, 500)

lines = {}
for i, v in feature_sliders.items():
    lines[i] = ax.plot(x, np.sin(v.value * x))[0]

fig2, ax2 = plt.subplots()
lines2 = ax2.bar(list(feature_sliders.keys()), [v.value for v in feature_sliders.values()])

counter = []

def update_new_graph(b):
    counter.append(1)
    for i, v in feature_sliders.items():
        lines[i].set_data(x, np.sin(v.value * x))
    fig.canvas.draw()
    fig.canvas.flush_events()

    ax2.clear()
    ax2.bar(list(feature_sliders.keys()), [v.value for v in feature_sliders.values()])

    fig2.canvas.draw()
    fig2.canvas.flush_events()
    counter_widget.value = format_count(len(counter))

run_button = wd.Button(description="=>", layout=wd.Layout(height='50px', width='50px'))
run_button.on_click(update_new_graph)


main = wd.HBox([
    slider_box,
    wd.VBox([run_button, counter_widget]),
    wd.VBox([
        fig.canvas,
        fig2.canvas
    ]),
])
