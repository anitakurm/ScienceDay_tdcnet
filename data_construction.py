# Databricks notebook source
import numpy as np
from matplotlib.pylab import plot
import pandas as pd
from sklearn import preprocessing
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
# MAGIC
# MAGIC ## Create Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical Features

# COMMAND ----------

 #https://www.wilsonamplifiers.com/blog/11-major-building-materials-that-kill-your-cell-phone-reception/
 #https://www.visualcapitalist.com/the-worlds-most-used-apps-by-downstream-traffic/

# COMMAND ----------


cat_variables = {
  'Occupation' : {
    'names' : ['Data Scientist','Teacher','Student','Nurse','Firefighter','Salesman'],
    'values' : [6,5,4,3,2,1]
    },
  'Family Type' : {
    'names' : ['Family','Couple','Singles','Seniors'],
    'values': [4,3,2,1]
  },
  'Family Interest' : {
    'names' :  ['Gaming','Musical','Crafty','Sporty','Travelling','Outdoor'],
    'values': [6,5,4,3,2,1]
  },
  'House Type' : {
    'names' : ['Metal','Glas','Concrete','Tiles','Wood'],
    'values': [5,4,3,2,1]
  },
  'Geo Type' : {
    'names' :['Urban','Suburban','Rural'],
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

numerical_features = ['Number Of Residents', 'Average Age', 'Distance To Nearest Tower [m]',
                      'Number Of Phones','Number Of Computers','Number Of Tvs', 'Number Of Pets'
                      ,'Customer Happiness',
                      'Time Spend On YouTube [min]', 'Time Spend On TikTok [min]', 
                      'Time Spend On Instagram [min]','Time Spend On Spotify [min]',
                      'Size of home [m2]', 
                    ]

# COMMAND ----------


# numerical_features = {
#    'Number Of Residents':{
#      'mean':1, 
#      'std': ,
#      'distribution': 'normal' 
#      },
#    'Average Age':{
#      'mean':1, 
#      'std': ,
#      'distribution': 'normal' 
#      },
#    'Distance To Nearest Tower':{
#      'mean':1, 
#      'std': ,
#      'distribution': 'normal' 
#      },
#    'Number Of Phones':{
#      'mean':1, 
#      'std': ,
#      'distribution': 'normal' 
#      },
#    'Number Of Computers':{
#      'mean':1, 
#      'std': ,
#      'distribution': 'normal' 
#      },
#    'Number Of Tvs':{
#      'mean':1, 
#      'std': ,
#      'distribution': 'normal' 
#      },
#    'Customer Happiness':{
#      'mean':1, 
#      'std': ,
#      'distribution': 'normal' 
#      },
#    'Time Spend On YouTube':{
#      'mean':1, 
#      'std': ,
#      'distribution': 'normal' 
#      },
#    'Time Spend On TikTok':{
#      'mean':1, 
#      'std': ,
#      'distribution': 'normal' 
#      },
#    'Time Spend On Instagram':{
#      'mean':1, 
#      'std': ,
#      'distribution': 'normal' 
#      },
#    'Time Spend On Spotify':{
#      'mean':1, 
#      'std': ,
#      'distribution': 'normal' 
#      },
#  }                   

# COMMAND ----------

# total_features = list(numerical_features.keys())+list(cat_variables.keys())
# total_features = ['Number Of Residents','Time Spend On YouTube','Occupation','Mobile Traffic']
total_features = cat_features + numerical_features + ['Mobile Traffic']

# COMMAND ----------

for i in range(0,len(total_features)):
  for j in range(0,len(total_features)):
    if  (i>j):
      print(f"'{total_features[i]} - {total_features[j]}' : 0. ,")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Update correlations

# COMMAND ----------

correlations_dict = {'Family Type - Occupation' : 0.6 ,
'Family Interest - Occupation' : 0 ,
'Family Interest - Family Type' : 0.5 ,
'House Type - Occupation' : 0.4 ,
'House Type - Family Type' : 0.4 ,
'House Type - Family Interest' : 0 ,
'Number Of Residents - Occupation' : 0 ,
'Number Of Residents - Family Type' : 0 ,
'Number Of Residents - Family Interest' : 0 ,
'Number Of Residents - House Type' : 0 ,
'Average Age - Occupation' : 0 ,
'Average Age - Family Type' : -0.7 ,
'Average Age - Family Interest' : 0 ,
'Average Age - House Type' : 0 ,
'Average Age - Number Of Residents' : 0.3 ,
'Distance To Nearest Tower [m] - Occupation' : 0 ,
'Distance To Nearest Tower [m] - Family Type' : 0 ,
'Distance To Nearest Tower [m] - Family Interest' : 0 ,
'Distance To Nearest Tower [m] - House Type' : 0 ,
'Distance To Nearest Tower [m] - Number Of Residents' : 0 ,
'Distance To Nearest Tower [m] - Average Age' : 0.4 ,
'Number Of Phones - Occupation' : 0.5 ,
'Number Of Phones - Family Type' : 0 ,
'Number Of Phones - Family Interest' : 0 ,
'Number Of Phones - House Type' : 0 ,
'Number Of Phones - Number Of Residents' : 0.8 ,
'Number Of Phones - Average Age' : 0.5 ,
'Number Of Phones - Distance To Nearest Tower [m]' : 0 ,
'Number Of Computers - Occupation' : 0.3 ,
'Number Of Computers - Family Type' : 0.3 ,
'Number Of Computers - Family Interest' : 0.3 ,
'Number Of Computers - House Type' : 0 ,
'Number Of Computers - Number Of Residents' : 0.8 ,
'Number Of Computers - Average Age' : 0.5 ,
'Number Of Computers - Distance To Nearest Tower [m]' : 0 ,
'Number Of Computers - Number Of Phones' : 0.7 ,
'Number Of Tvs - Occupation' : 0 ,
'Number Of Tvs - Family Type' : 0 ,
'Number Of Tvs - Family Interest' : 0.5 ,
'Number Of Tvs - House Type' : 0 ,
'Number Of Tvs - Number Of Residents' : 0.7 ,
'Number Of Tvs - Average Age' : 0.5 ,
'Number Of Tvs - Distance To Nearest Tower [m]' : 0 ,
'Number Of Tvs - Number Of Phones' : 0.5 ,
'Number Of Tvs - Number Of Computers' : 0.5 ,
'Number Of Pets - Occupation' : 0 ,
'Number Of Pets - Family Type' : 0 ,
'Number Of Pets - Family Interest' : 0 ,
'Number Of Pets - House Type' : 0 ,
'Number Of Pets - Number Of Residents' : 0. ,
'Number Of Pets - Average Age' : 0. ,
'Number Of Pets - Distance To Nearest Tower [m]' : 0. ,
'Number Of Pets - Number Of Phones' : 0. ,
'Number Of Pets - Number Of Computers' : 0. ,
'Number Of Pets - Number Of Tvs' : 0. ,
'Customer Happiness - Occupation' : 0. ,
'Customer Happiness - Family Type' : 0. ,
'Customer Happiness - Family Interest' : 0. ,
'Customer Happiness - House Type' : 0. ,
'Customer Happiness - Number Of Residents' : 0. ,
'Customer Happiness - Average Age' : 0. ,
'Customer Happiness - Distance To Nearest Tower [m]' : -0.7 ,
'Customer Happiness - Number Of Phones' : 0. ,
'Customer Happiness - Number Of Computers' : 0. ,
'Customer Happiness - Number Of Tvs' : 0. ,
'Customer Happiness - Number Of Pets' : 0. ,
'Time Spend On YouTube [min] - Occupation' : 0.5 ,
'Time Spend On YouTube [min] - Family Type' : 0.5 ,
'Time Spend On YouTube [min] - Family Interest' : 0.5 ,
'Time Spend On YouTube [min] - House Type' : 0. ,
'Time Spend On YouTube [min] - Number Of Residents' : 0.6 ,
'Time Spend On YouTube [min] - Average Age' : -0.4 ,
'Time Spend On YouTube [min] - Distance To Nearest Tower [m]' : 0. ,
'Time Spend On YouTube [min] - Number Of Phones' : 0.5 ,
'Time Spend On YouTube [min] - Number Of Computers' : 0.5 ,
'Time Spend On YouTube [min] - Number Of Tvs' : 0. ,
'Time Spend On YouTube [min] - Number Of Pets' : 0. ,
'Time Spend On YouTube [min] - Customer Happiness' : 0. ,
'Time Spend On TikTok [min] - Occupation' : 0. ,
'Time Spend On TikTok [min] - Family Type' : 0.5 ,
'Time Spend On TikTok [min] - Family Interest' : 0.5 ,
'Time Spend On TikTok [min] - House Type' : 0. ,
'Time Spend On TikTok [min] - Number Of Residents' : 0.5 ,
'Time Spend On TikTok [min] - Average Age' : -0.7 ,
'Time Spend On TikTok [min] - Distance To Nearest Tower [m]' : 0. ,
'Time Spend On TikTok [min] - Number Of Phones' : 0.6 ,
'Time Spend On TikTok [min] - Number Of Computers' : 0. ,
'Time Spend On TikTok [min] - Number Of Tvs' : 0. ,
'Time Spend On TikTok [min] - Number Of Pets' : 0. ,
'Time Spend On TikTok [min] - Customer Happiness' : 0. ,
'Time Spend On TikTok [min] - Time Spend On YouTube [min]' : 0.3 ,
'Time Spend On Instagram [min] - Occupation' : 0. ,
'Time Spend On Instagram [min] - Family Type' : 0.5 ,
'Time Spend On Instagram [min] - Family Interest' : 0.3 ,
'Time Spend On Instagram [min] - House Type' : 0. ,
'Time Spend On Instagram [min] - Number Of Residents' : 0.3 ,
'Time Spend On Instagram [min] - Average Age' : -0.3 ,
'Time Spend On Instagram [min] - Distance To Nearest Tower [m]' : 0. ,
'Time Spend On Instagram [min] - Number Of Phones' : 0.6 ,
'Time Spend On Instagram [min] - Number Of Computers' : 0. ,
'Time Spend On Instagram [min] - Number Of Tvs' : 0. ,
'Time Spend On Instagram [min] - Number Of Pets' : 0. ,
'Time Spend On Instagram [min] - Customer Happiness' : 0. ,
'Time Spend On Instagram [min] - Time Spend On YouTube [min]' : 0.3 ,
'Time Spend On Instagram [min] - Time Spend On TikTok [min]' : 0.5 ,
'Time Spend On Spotify [min] - Occupation' : 0. ,
'Time Spend On Spotify [min] - Family Type' : 0. ,
'Time Spend On Spotify [min] - Family Interest' : 0. ,
'Time Spend On Spotify [min] - House Type' : 0. ,
'Time Spend On Spotify [min] - Number Of Residents' : 0.5 ,
'Time Spend On Spotify [min] - Average Age' : -0.2 ,
'Time Spend On Spotify [min] - Distance To Nearest Tower [m]' : 0. ,
'Time Spend On Spotify [min] - Number Of Phones' : 0.5 ,
'Time Spend On Spotify [min] - Number Of Computers' : 0.5 ,
'Time Spend On Spotify [min] - Number Of Tvs' : 0. ,
'Time Spend On Spotify [min] - Number Of Pets' : 0. ,
'Time Spend On Spotify [min] - Customer Happiness' : 0. ,
'Time Spend On Spotify [min] - Time Spend On YouTube [min]' : 0.5 ,
'Time Spend On Spotify [min] - Time Spend On TikTok [min]' : 0.5 ,
'Time Spend On Spotify [min] - Time Spend On Instagram [min]' : 0.5 ,
'Size of home [m2] - Occupation' : 0.6 ,
'Size of home [m2] - Family Type' : 0.6 ,
'Size of home [m2] - Family Interest' : 0. ,
'Size of home [m2] - House Type' : 0. ,
'Size of home [m2] - Number Of Residents' : 0.7 ,
'Size of home [m2] - Average Age' : 0.3 ,
'Size of home [m2] - Distance To Nearest Tower [m]' : 0.6 ,
'Size of home [m2] - Number Of Phones' : 0.3 ,
'Size of home [m2] - Number Of Computers' : 0.3 ,
'Size of home [m2] - Number Of Tvs' : 0.3 ,
'Size of home [m2] - Number Of Pets' : 0.6 ,
'Size of home [m2] - Customer Happiness' : 0. ,
'Size of home [m2] - Time Spend On YouTube [min]' : 0. ,
'Size of home [m2] - Time Spend On TikTok [min]' : 0. ,
'Size of home [m2] - Time Spend On Instagram [min]' : 0. ,
'Size of home [m2] - Time Spend On Spotify [min]' : 0. ,
'Mobile Traffic - Occupation' : 0.3 ,
'Mobile Traffic - Family Type' : 0.3 ,
'Mobile Traffic - Family Interest' : 0.5 ,
'Mobile Traffic - House Type' : 0.0 ,
'Mobile Traffic - Number Of Residents' : 0.6 ,
'Mobile Traffic - Average Age' : -0.4 ,
'Mobile Traffic - Distance To Nearest Tower [m]' : -0.6 ,
'Mobile Traffic - Number Of Phones' : 0.6 ,
'Mobile Traffic - Number Of Computers' : 0.3 ,
'Mobile Traffic - Number Of Tvs' : 0. ,
'Mobile Traffic - Number Of Pets' : 0. ,
'Mobile Traffic - Customer Happiness' : 0.6 ,
'Mobile Traffic - Time Spend On YouTube [min]' : 0.6 ,
'Mobile Traffic - Time Spend On TikTok [min]' : 0.3 ,
'Mobile Traffic - Time Spend On Instagram [min]' : 0.3 ,
'Mobile Traffic - Time Spend On Spotify [min]' : 0.1 ,
'Mobile Traffic - Size of home [m2]' : 0.3 }

# COMMAND ----------

def random_correlation(max_random=0.05):
  return np.random.random()*max_random-max_random/2


cov_matrix = np.eye(len(total_features))

variables = {x:i for i,x in enumerate(total_features)}

correlations = correlations_dict

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

correlated = np.random.multivariate_normal([1 for x in total_features], cov_matrix, size=12000)

# COMMAND ----------

#Scale data to range [0,1]
x = correlated 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df.columns = total_features


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Transform numeric values
# MAGIC

# COMMAND ----------

df['Number Of Residents'] = np.floor(np.exp(df['Number Of Residents']*2)).astype(int)
df['Average Age'] = (df['Average Age']*60 + 20).astype(int)
df['Distance To Nearest Tower [m]'] = (np.random.lognormal(0.55,0.2, 12000) * df['Mobile Traffic'] * df['Distance To Nearest Tower [m]'])*150
#df['Distance To Nearest Tower [m]'] = np.random.lognormal(0.65,0.28, 12000)  * df['Mobile Traffic']
df['Number Of Phones'] =(np.floor(df['Number Of Phones'] * 6)).astype(int)
df['Number Of Computers'] = (np.floor(df['Number Of Computers'] * 3.3)).astype(int)
df['Number Of Tvs'] = (np.floor(df['Number Of Tvs'] * 3.4)).astype(int)
df['Number Of Pets'] = (np.floor(df['Number Of Pets'] * 2.5)).astype(int)
df['Time Spend On YouTube [min]'] = np.exp(df['Time Spend On YouTube [min]']*4) +np.random.randint(10, size=12000) 
df['Time Spend On TikTok [min]'] = np.exp(df['Time Spend On TikTok [min]']*4.5)
df['Time Spend On Instagram [min]'] = np.exp(df['Time Spend On Instagram [min]']*5)
df['Time Spend On Spotify [min]'] = np.exp(df['Time Spend On Spotify [min]']*6)
df['Mobile Traffic'] = df['Mobile Traffic']*20
df['Customer Happiness'] = pd.cut(df['Customer Happiness'] ,[0,0.2,0.21,0.22,0.3,0.35,0.4,0.5,0.65,0.7,1], labels = [1,2,3,4,5,6,7,8,9,10])

# COMMAND ----------

pd.Series(np.random.default_rng().exponential(scale=4, size=100)).hist()

# COMMAND ----------

import matplotlib.pyplot as plt
from scipy.stats import lognorm

# r = lognorm.rvs(1.1, size=1200) * df['Mobile Traffic']
# r = (np.random.lognormal(0.55,0.2, 1200) * df['Mobile Traffic'] * df['Distance To Nearest Tower [m]'])*150

plt.scatter( df['Distance To Nearest Tower [m]'] , df['Mobile Traffic'])
plt.show()

# COMMAND ----------

df['Customer Happiness'] = df['Customer Happiness'].fillna(8)

# COMMAND ----------

#export
df_raw = df.copy()
df.to_csv('data/transformed_data_raw.csv')


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Transform data for visualisation

# COMMAND ----------

# Transform Categorical features from numeric to discrete
for i in cat_features:
  bins= df[i].quantile(np.linspace(0,1,len(cat_variables[i]['values']) + 1))
  df[i] = pd.cut(df[i],bins, labels=cat_variables[i]['names'])

df.to_csv('transformed_data_for_viewing.csv')


# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Test Modelling

# COMMAND ----------

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# COMMAND ----------

X = df_raw.values[:,0:(I-1)]
Y = df_raw.values[:,(I-1):]

X_train, X_test, y_train, y_test = train_test_split(X,Y ,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)

# COMMAND ----------



# COMMAND ----------

lin = LinearRegression().fit(X_train,y_train)
y_train_predict = lin.predict(X_train)
y_test_predict = lin.predict(X_test)

# COMMAND ----------

from matplotlib.pylab import plot

# COMMAND ----------

plot(y_train,y_train_predict,'.')
plot(np.linspace(0,15,10),np.linspace(0,15,10))

# COMMAND ----------

# Finalize parameters and transformations
# Create plots
# Create models
# Widgets for testing
# 1 page for linear regression manual weights
# 1 page for linear regression
# 
# Prep output
# Live submits to track performence LEADERBOARD 
# Notebook of "How to "

# COMMAND ----------


