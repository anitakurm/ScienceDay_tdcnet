# Databricks notebook source
import numpy as np
# print features and target


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Price of product
# MAGIC - Distance to tower
# MAGIC - 2G Speed
# MAGIC - 3G speed
# MAGIC - 4G speed
# MAGIC - 5G speed
# MAGIC - Number of residents
# MAGIC - Number of Phones
# MAGIC - Customer happiness (Customer experience, NPS)
# MAGIC - Tvs in household 
# MAGIC - Number of Microwaves
# MAGIC - Number of Pets
# MAGIC - Family type (youngsters, seniors families small kids, families w. big kids)
# MAGIC - Family interest (Sporty (watches a lot of sports), outdoorsy, gaming, cooking)
# MAGIC - Occupation (Desk work, Nurse etc. who uses internet) 
# MAGIC - MDU/SDU (MDUs block signal) 
# MAGIC - Concrete, wood, brick building (obstacles lower speed) 
# MAGIC - Time spend on tiktok, instragram, netflix etc ? Maybe some apps generate less traffic vs usage compared to others
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical

# COMMAND ----------

 #https://www.wilsonamplifiers.com/blog/11-major-building-materials-that-kill-your-cell-phone-reception/
 #https://www.visualcapitalist.com/the-worlds-most-used-apps-by-downstream-traffic/

# COMMAND ----------


cat_variables = {
  'Occupation' : {
    'names' : ['Data Scientist','Teacher','Nurse','Firefighter'],
    'values' : [5,4,3,2,1]
    },
  'Family Type' : {
    'names' : ['Family','Couple','Singles','Seniors'],
    'values': [4,3,2,1]
  },
  'Family Interest' : {
    'names' :  ['Gaming','Sporty','Outdoor'],
    'values': [3,2,1]
  },
  'House Type' : {
    'names' : ['Metal','Glas','Concrete','Tiles','Wood'],
    'values': [5,4,3,2,1]
  },
  'Geo Type' : {
    'names' :['city','Suburban','rural'],
    'values': [3,2,1]
  }
}


# COMMAND ----------

cat_features = ['Occupation','Family Type','Family Interest','House Type']

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Numeric features

# COMMAND ----------

numerical_features = ['Number Of Residents', 'Average Age', 'Distance To Nearest Tower',
                      'Number Of Phones','Number Of Computers','Number Of Tvs','Customer Happiness',
                     'Time Spend On YouTube', 'Time Spend On TikTok', 'Time Spend On Instagram','Time Spend On Spotify']
numerical_features = {
   'Number Of Residents':{
     'mean':1, 
     'std': ,
     'distribution': 'normal' 
     },
   'Average Age':{
     'mean':1, 
     'std': ,
     'distribution': 'normal' 
     },
   'Distance To Nearest Tower':{
     'mean':1, 
     'std': ,
     'distribution': 'normal' 
     },
   'Number Of Phones':{
     'mean':1, 
     'std': ,
     'distribution': 'normal' 
     },
   'Number Of Computers':{
     'mean':1, 
     'std': ,
     'distribution': 'normal' 
     },
   'Number Of Tvs':{
     'mean':1, 
     'std': ,
     'distribution': 'normal' 
     },
   'Customer Happiness':{
     'mean':1, 
     'std': ,
     'distribution': 'normal' 
     },
   'Time Spend On YouTube':{
     'mean':1, 
     'std': ,
     'distribution': 'normal' 
     },
   'Time Spend On TikTok':{
     'mean':1, 
     'std': ,
     'distribution': 'normal' 
     },
   'Time Spend On Instagram':{
     'mean':1, 
     'std': ,
     'distribution': 'normal' 
     },
   'Time Spend On Spotify':{
     'mean':1, 
     'std': ,
     'distribution': 'normal' 
     },
 }                   

# COMMAND ----------

# total_features = list(numerical_features.keys())+list(cat_variables.keys())
total_features = ['Number Of Residents','Time Spend On YouTube','Occupation','Mobile Traffic']

# COMMAND ----------

for i in range(0,len(total_features)):
  for j in range(0,len(total_features)):
    if  (i>j):
      print(f"'{total_features[i]} - {total_features[j]}' : 0. ,")


# COMMAND ----------

def random_correlation(max_random=0.05):
  return np.random.random()*max_random-max_random/2


cov_matrix = np.eye(len(total_features))

variables = {x:i for i,x in enumerate(total_features)}

correlations = {
'Time Spend On YouTube - Number Of Residents' : 0.5 ,
'Occupation - Number Of Residents' : 0.1 ,
'Occupation - Time Spend On YouTube' : 0.4 ,
'Mobile Traffic - Number Of Residents' : 0.6 ,
'Mobile Traffic - Time Spend On YouTube' : 0.5 ,
'Mobile Traffic - Occupation' : 0.3 
}

for corre in correlations:
  number = correlations[corre]
  x,y = corre.split(' - ')
  if variables[x]>variables[y]:
    cov_matrix[variables[y],variables[x]] = number
  else:
    cov_matrix[variables[x],variables[y]] = number

I,J = cov_matrix.shape
for i in range(I):
  for j in range(J):
    if i==j:
      continue
    
    if i>j:
      
      if cov_matrix[j,i]==0:
        cov_matrix[j,i] = random_correlation()

      cov_matrix[i,j] = cov_matrix[j,i]

# COMMAND ----------

correlated = np.random.multivariate_normal([1 for x in total_features], cov_matrix, size=1000)

# COMMAND ----------

import pandas as pd
from sklearn import preprocessing
x = correlated 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df.columns = total_features
df.head()

# COMMAND ----------

df['Number Of Residents'] = np.floor(np.exp(df['Number Of Residents']*2)).astype(int)
df['Time Spend On YouTube'] = np.exp(df['Time Spend On YouTube']*5.5)
bins= df['Occupation'].quantile(np.linspace(0,1,len(cat_variables['Occupation']['values'])))
df['Occupation'] = pd.cut(df['Occupation'],bins, labels=cat_variables['Occupation']['names'])
df['Mobile Traffic'] = np.floor(np.exp(df['Mobile Traffic']*4))

# COMMAND ----------

df.hist()

# COMMAND ----------

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# COMMAND ----------

X = correlated[:,0:(I-1)]
Y = correlated[:,(I-1):]

X_train, X_test, y_train, y_test = train_test_split(X,Y ,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)

# COMMAND ----------

lin = LinearRegression().fit(X_train,y_train)
y_train_predict = lin.predict(X_train)
y_test_predict = lin.predict(X_test)

# COMMAND ----------

from matplotlib.pylab import plot

# COMMAND ----------

plot(y_test,y_test_predict,'.')

# COMMAND ----------

plot(y_train,y_train_predict,'.')

# COMMAND ----------


