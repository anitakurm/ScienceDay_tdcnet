# Databricks notebook source
import ipywidgets as widgets

# COMMAND ----------

# MAGIC %md
# MAGIC # Slider

# COMMAND ----------

slider = widgets.IntSlider(
    min=0,
    max=10,
    step=1,
    description='Slider:',
    value=3
)

display(slider)

# COMMAND ----------

slider.value

# COMMAND ----------

# MAGIC %md
# MAGIC # Button doesn't work :( 

# COMMAND ----------

btn = widgets.Button(description='Medium')
display(btn)
def btn_eventhandler(obj):
    print('Hello from the {} button!'.format(obj.description))
btn.on_click(btn_eventhandler)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Checkbox - singular

# COMMAND ----------

check = widgets.Checkbox(
    value=False,
    description='Check me',
    disabled=False,
    indent=False
)

check2 = widgets.Checkbox(
    value=False,
    description='Check me too',
    disabled=False,
    indent=False
)

display(check)
display(check2)

# COMMAND ----------

check.value

# COMMAND ----------

check2.value

# COMMAND ----------

# MAGIC %md
# MAGIC # Select multiple with ctrl

# COMMAND ----------

select = widgets.SelectMultiple(
    options=['Apples', 'Oranges', 'Pears'],
    value=['Oranges'],
    #rows=10,
    description='Fruits',
    disabled=False
)

# COMMAND ----------

display(select)

# COMMAND ----------

select.value

# COMMAND ----------

# MAGIC %md 
# MAGIC # POC feature selection widget
# MAGIC Select which features are selected in the df

# COMMAND ----------

import pandas as pd
import numpy as np

df = pd.read_csv("/dbfs/FileStore/shared_uploads/kurm@tdcnet.dk/internet_session.csv")

# COMMAND ----------



# COMMAND ----------

dropdown = widgets.SelectMultiple(options = df.columns)

def dropdown_eventhandler(change):
    if (change.new == ALL):
        display(df)
    else:
        display(df[change.new])


dropdown.observe(dropdown_eventhandler, names='value')
display(dropdown)

# COMMAND ----------

#dropdown = widgets.Dropdown(options = df.columns)
dropdown = widgets.SelectMultiple(options = df.columns)
output_feature_select = widgets.Output()

def dropdown_eventhandler(change):
    output_feature_select.clear_output()
    with output_feature_select:
        
        display(df[list(change.new)])


dropdown.observe(dropdown_eventhandler, names='value')


display(dropdown)

# COMMAND ----------

display(output_feature_select)

# COMMAND ----------

# Helper functions
ALL = 'ALL'
def dropdown_eventhandler(change):
    if (change.new == ALL):
        display(df)
    else:
        display(df[change.new])

# COMMAND ----------

dropdown.observe(dropdown_eventhandler, names='value')

# COMMAND ----------

df.columns

# COMMAND ----------

df[['name', 'start_time']]

# COMMAND ----------


