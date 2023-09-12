# Databricks notebook source
import pandas as pd
import ipywidgets as widgets
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# COMMAND ----------

# Read Data
df_raw = pd.read_csv('/Workspace/Repos/sall@tdcnet.dk/ScienceDay_tdcnet/transformed_data_raw.csv')

# COMMAND ----------

# Read Data
df_view = pd.read_csv('/Workspace/Repos/sall@tdcnet.dk/ScienceDay_tdcnet/transformed_data_for_viewing.csv')

# COMMAND ----------

# Hide last 200 observations for testing 
df_unseen = df_raw[-200:].reset_index()
df = df_raw[:-200].reset_index()

# COMMAND ----------

df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('\[', '')
df.columns = df.columns.str.replace('\]', '')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Make sliders

# COMMAND ----------

slider_min = -10
slider_max = 10
slider_value = (slider_max+slider_min)/2

# COMMAND ----------

#Specify variables
variables =  ['Occupation', 'Family_Type', 'Family_Interest',
       'House_Type', 'Number_Of_Residents', 'Average_Age',
       'Distance_To_Nearest_Tower_m', 'Number_Of_Phones',
       'Number_Of_Computers', 'Number_Of_Tvs', 'Number_Of_Pets',
       'Customer_Happiness', 'Time_Spend_On_YouTube_min',
       'Time_Spend_On_TikTok_min', 'Time_Spend_On_Instagram_min',
       'Time_Spend_On_Spotify_min', 'Size_of_home_m2']
# df.columns


# COMMAND ----------

layout = widgets.Layout(width='auto', height='40px') #set width and height


# COMMAND ----------

for i in variables:
  new_name =  i.replace('_',' ')
  globals()[f'slider_{i}'] = slider_Occupation = widgets.IntSlider(
                                    min=slider_min,
                                    max=slider_max,
                                    step=1,
                                    description=new_name + ': ',
                                    value=slider_value,
                                    layout=widgets.Layout(width='40%'),
                                    style= {'description_width': '40%'}
                                     )

# COMMAND ----------

for i in variables: 
  new_name =  i.replace(' ','_')
  display(globals()[f'slider_{new_name}'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Store sliders in Dataframe

# COMMAND ----------

# button = widgets.Button(description="Save weights")
# output = widgets.Output()

# display(button, output)

# def on_button_clicked(b):
#   #Get slider values as pandas df
#   df_weights = pd.DataFrame(columns=['variable', 'weight'])
#   counter = 0
#   for i in variables: 
#     new_name =  i.replace(' ','_')
#     df_weights.loc[counter,'variable'] = i
#     df_weights.loc[counter,'weight']= globals()[f'slider_{new_name}'].value
#     counter +=1 

#   with output:
#       print("Weights Saved")

# button.on_click(on_button_clicked)

# COMMAND ----------

# Add save button for weights and store data
get_data_button = widgets.Button(description='Save Weights')
output = widgets.Output()

def get_data(b):

  #Get slider values as pandas df
  tmp_weights = pd.DataFrame(columns=['variable', 'weight'])
  counter = 0
  with output:
    print("Weights Saved")

  for i in variables: 
    new_name =  i.replace(' ','_')
    tmp_weights.loc[counter,'variable'] = i
    tmp_weights.loc[counter,'weight']= globals()[f'slider_{new_name}'].value
    counter +=1 
    get_data.data = tmp_weights

  print(get_data.data)

  return get_data.data

# DISPLAY BUTTON
get_data_button.on_click(get_data)
display(get_data_button, output)


# COMMAND ----------

# Store weights as dataframe
df_weights = get_data.data
df_weights

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create manual linear regression using user input

# COMMAND ----------

# Normalize weights between -1 and 1
df_weights['weight'] = df_weights['weight']/10

# COMMAND ----------

# standardizing dataframe so coefficients are -1 and 1
df_z = df.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)

# COMMAND ----------

# Store mean and std to transform back 
mean_std={}
for var in df.columns:
    mean_std[var]=(df[var].mean(), df[var].std())

# COMMAND ----------

def reverse_zscore(pandas_series, mean, std):
    '''Mean and standard deviation should be of original variable before standardization'''
    yis=pandas_series*std+mean
    return yis

var = 'Mobile_Traffic'
original_mean, original_std = mean_std[var]
original_var_series = reverse_zscore(df_z[var], original_mean, original_std)

# COMMAND ----------


# Manually calculate y = alpha * x + beta
Y = df_z['Mobile_Traffic']
X = df_z[variables].copy()
for i in variables:
  alpha = df_weights[df_weights['variable'] == i].weight.values[0]
  X[i] = df_z[i] * alpha 
Y_pred = X.sum(axis=1)

# COMMAND ----------

# plot
import matplotlib.pyplot as plt

Y_actual = df['Mobile_Traffic']
Y_pred_trans = reverse_zscore(Y_pred, original_mean, original_std)
plt.scatter(Y_pred_trans,Y_actual)
plt.plot([0, max(max(Y_actual), max(Y_pred_trans))], [0,max(max(Y_actual), max(Y_pred_trans))], 'red')
plt.title("Mobile Traffic - Manual Model")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make Linear Regression

# COMMAND ----------

#Create formula
# Remove Intercept
formula ='Mobile_Traffic ~ -1 + '
counter = 1
for i in  X.columns.values:
  if counter == 1:
    formula = formula + i
  else:
    formula = formula + ' + ' + i
  counter += 1


# COMMAND ----------

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
res = smf.ols(formula, data=df_z).fit()
lm_prediction = res.predict(df_z[:])
res.summary()


# COMMAND ----------

# plot
import matplotlib.pyplot as plt

lm_prediction_trans = reverse_zscore(lm_prediction, original_mean, original_std)

plt.scatter(lm_prediction_trans, Y_actual)
plt.plot([0, max(max(Y_actual), max(lm_prediction_trans))], [0,max(max(Y_actual), max(lm_prediction_trans))], 'red')
plt.title("Mobile Traffic - Linear Model")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


# COMMAND ----------

# !pip install ipympl
# %matplotlib ipympl

# COMMAND ----------

# %matplotlib inline  
# %matplotlib ipympl
# %pylab

# COMMAND ----------

# Show case Non Linear relationship
plt.scatter(df['Distance_To_Nearest_Tower_m'], Y_actual)
plt.title("Mobile Traffic - Linear Model")
plt.ylabel("Mobile Traffic")
plt.xlabel("-")
plt.show()


# COMMAND ----------

# Example 3d plot
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

model = smf.ols(formula='Mobile_Traffic ~ Average_Age + Number_Of_Phones', data=df)

results = model.fit()
x, y = model.exog_names[1:]

x_range = np.arange(df[x].min(), df[x].max())
print(x_range.shape)
y_range = np.arange(df[y].min(), df[y].max())
print(y_range.shape)

X, Y = np.meshgrid(x_range, y_range)

exog = pd.DataFrame({x: X.ravel(), y: Y.ravel()})
Z = results.predict(exog = exog).values.reshape(X.shape)
y_pred = results.predict(df[model.exog_names[1:]])

fig = plt.figure(figsize=plt.figaspect(1)*2)
ax = plt.axes(projection='3d')
ax.scatter(df[x].values, df[y].values, df[model.endog_names].values, label="Actual")
ax.scatter(df[x].values, df[y].values, y_pred, label="Pred")

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha = 0.4)
ax.legend()
plt.show()

# COMMAND ----------

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import matplotlib.pyplot as plt
# import numpy as np

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)

# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()
