# Databricks notebook source
from sklearn.datasets import make_regression

# create features and targets
features, target = make_regression(n_samples=100,
                                  n_features=10,
                                  n_informative=5,
                                  n_targets=1,
                                  random_state=42)

# print features and target
print("Features:")
print(features[:5])
print("Target:")
print(target[:5])

# COMMAND ----------



# COMMAND ----------

