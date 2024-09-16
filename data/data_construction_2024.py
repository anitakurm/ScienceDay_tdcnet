import numpy as np
from matplotlib.pylab import plot
import pandas as pd
from sklearn import preprocessing
from scipy.stats import multivariate_normal as mvn
import seaborn as sns

np.random.seed(seed=44)


cat_variables = {
    "Occupation": {
        "names": ["Data Scientist", "Student", "Teacher", "Nurse", "Firefighter"],
        "values": [6, 5, 4, 3, 2, 1],
    },
    "Family Type": {
        "names": ["Family", "Couple", "Singles", "Seniors"],
        "values": [4, 3, 2, 1],
    },
    "Family Interest": {
        "names": ["Gaming", "Musical", "Crafty", "Sporty", "Travelling", "Outdoor"],
        "values": [6, 5, 4, 3, 2, 1],
    },
    "House Type": {
        "names": ["Metal", "Glas", "Concrete", "Tiles", "Wood"],
        "values": [5, 4, 3, 2, 1],
    },
    "Geo Type": {"names": ["Urban", "Suburban", "Rural"], "values": [3, 2, 1]},
}

cat_features = ["Occupation", "Family Type", "Family Interest", "House Type"]


numerical_features = [
    "Number Of Residents",
    "Average Age",
    "Distance To Nearest Tower [m]",
    "Number Of Phones",
    "Number Of Computers",
    "Number Of Tvs",
    "Number Of Pets",
    "Customer Happiness",
    "Time Spend On YouTube [min]",
    "Time Spend On TikTok [min]",
    "Time Spend On Instagram [min]",
    "Time Spend On Spotify [min]",
    "Size of home [m2]",
]

min_max_scaler = preprocessing.MinMaxScaler()
n_samples = 1200


data = {
    "Mobile Traffic": [1, 0.6, 0.5, 0.3, 0.1, 0.4],
    "Time Spend On YouTube [min]": [0.6, 1, 0.1, 0.01, 0.01, 0.01],
    "Time Spend On TikTok [min]": [0.5, 0.1, 1, 0.01, 0.01, 0.01],
    "Time Spend On Instagram [min]": [0.3, 0.01, 0.01, 1, 0.01, 0.01],
    "Time Spend On Spotify [min]": [0.1, 0.01, 0.01, 0.01, 1, 0.01],
    "Size of home [m2]": [0.4, 0.01, 0.01, 0.01, 0.01, 1],
}

df = pd.DataFrame(data)
df.index = df.columns

# Ensure symmetry
matrix = (df + df.T) / 2
# Make the matrix positive semi-definite
eigenvalues, eigenvectors = np.linalg.eig(matrix.values)
positive_eigenvalues = np.maximum(eigenvalues, 0)
positive_matrix = np.dot(
    np.dot(eigenvectors, np.diag(positive_eigenvalues)), np.linalg.inv(eigenvectors)
)
matrix = pd.DataFrame(positive_matrix, index=matrix.index, columns=matrix.columns)
scores = mvn.rvs(mean=np.ones(len(matrix)), cov=matrix, size=n_samples)
# Scale data to range [0,1]
scores = min_max_scaler.fit_transform(scores)

final_df_1 = pd.DataFrame(scores)
final_df_1.columns = df.columns
final_df_1 = final_df_1.sort_values(by="Mobile Traffic")
final_df_1.reset_index(drop=True, inplace=True)


data = {
    "Mobile Traffic": [1, -0.3, 0.3, 0.4, 0],
    "Occupation": [-0.3, 1, 0.01, 0.01, 0.01],
    "Family Type": [0.3, 0.01, 1, 0.01, 0.01],
    "Family Interest": [0.4, 0.01, 0.01, 1, 0.01],
    "House Type": [0, 0.01, 0.01, 0.01, 1],
}

df = pd.DataFrame(data)
df.index = df.columns

# Ensure symmetry
matrix = (df + df.T) / 2
# Make the matrix positive semi-definite
eigenvalues, eigenvectors = np.linalg.eig(matrix.values)
positive_eigenvalues = np.maximum(eigenvalues, 0)
positive_matrix = np.dot(
    np.dot(eigenvectors, np.diag(positive_eigenvalues)), np.linalg.inv(eigenvectors)
)
matrix = pd.DataFrame(positive_matrix, index=matrix.index, columns=matrix.columns)
scores = mvn.rvs(mean=np.ones(len(matrix)), cov=matrix, size=n_samples)
# Scale data to range [0,1]
scores = min_max_scaler.fit_transform(scores)

final_df_2 = pd.DataFrame(scores)
final_df_2.columns = df.columns
final_df_2 = final_df_2.sort_values(by="Mobile Traffic")
final_df_2.reset_index(drop=True, inplace=True)


data = {
    "Mobile Traffic": [1, 0.7, -0.6, -0.5, 0.4, 0.3, 0.2, -0.3, 0.5],
    "Number Of Residents": [0.7, 1, 0.01, 0.01, 0.7, 0.5, 0.2, 0.01, 0.01],
    "Average Age": [-0.6, 0.01, 1, 0.01, 0.3, 0.3, 0.2, 0.01, 0.01],
    "Distance To Nearest Tower [m]": [
        -0.5,
        0.01,
        0.01,
        1,
        0.01,
        0.01,
        0.01,
        0.01,
        -0.7,
    ],
    "Number Of Phones": [0.4, 0.7, 0.3, 0.01, 1, 0.5, 0.2, 0.01, 0.01],
    "Number Of Computers": [0.3, 0.5, 0.3, 0.01, 0.5, 1, 0.2, 0.01, 0.01],
    "Number Of Tvs": [0.2, 0.2, 0.2, 0.01, 0.2, 0.2, 1, 0.01, 0.01],
    "Number Of Pets": [-0.3, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1, 0.01],
    "Customer Happiness": [0.5, 0.01, 0.01, -0.7, 0.01, 0.01, 0.01, 0.01, 1],
}

df = pd.DataFrame(data)
df.index = df.columns

# Ensure symmetry
matrix = (df + df.T) / 2
# Make the matrix positive semi-definite
eigenvalues, eigenvectors = np.linalg.eig(matrix.values)
positive_eigenvalues = np.maximum(eigenvalues, 0)
positive_matrix = np.dot(
    np.dot(eigenvectors, np.diag(positive_eigenvalues)), np.linalg.inv(eigenvectors)
)
matrix = pd.DataFrame(positive_matrix, index=matrix.index, columns=matrix.columns)
scores = mvn.rvs(mean=np.ones(len(matrix)), cov=matrix, size=n_samples)
# Scale data to range [0,1]
scores = min_max_scaler.fit_transform(scores)

final_df_3 = pd.DataFrame(scores)
final_df_3.columns = df.columns
final_df_3 = final_df_3.sort_values(by="Mobile Traffic")
final_df_3.reset_index(drop=True, inplace=True)

df = pd.concat(
    [
        final_df_1,
        final_df_2.drop(columns="Mobile Traffic"),
        final_df_3.drop(columns="Mobile Traffic"),
    ],
    axis=1,
)


# Transform
df["Number Of Residents"] = np.floor(np.exp(df["Number Of Residents"] * 2)).astype(int)
df["Average Age"] = (df["Average Age"] * 60 + 20).astype(int)
df["Distance To Nearest Tower [m]"] = (
    (np.random.lognormal(0.55, 0.2, n_samples) * (-1))
    * df["Mobile Traffic"]
    * df["Distance To Nearest Tower [m]"]
) * 1500 + 2000
df["Number Of Phones"] = (np.ceil(df["Number Of Phones"] * 5)).astype(int)
df["Number Of Computers"] = (np.ceil(df["Number Of Computers"] * 3.5)).astype(int)
df["Number Of Tvs"] = (np.ceil((df["Number Of Tvs"] * 3.8) - 0.5)).astype(int)
df["Number Of Pets"] = (np.ceil(df["Number Of Pets"] * 2.5)).astype(int)
df["Time Spend On YouTube [min]"] = np.exp(
    df["Time Spend On YouTube [min]"] * 4
) + np.random.randint(10, size=n_samples)
df["Time Spend On TikTok [min]"] = np.exp(df["Time Spend On TikTok [min]"] * 4.5)
df["Time Spend On Instagram [min]"] = np.exp(df["Time Spend On Instagram [min]"] * 5)
df["Time Spend On Spotify [min]"] = np.exp(df["Time Spend On Spotify [min]"] * 6)
df["Mobile Traffic"] = df["Mobile Traffic"] * 20 + 4
df["Customer Happiness"] = pd.cut(
    df["Customer Happiness"],
    [0, 0.2, 0.21, 0.22, 0.3, 0.35, 0.4, 0.5, 0.65, 0.7, 1],
    labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)
df["Size of home [m2]"] = df["Size of home [m2]"] * 200 + 35


# Bin categorical features
df["Customer Happiness"] = pd.to_numeric(df["Customer Happiness"], errors="coerce")
df["Customer Happiness"] = df["Customer Happiness"].fillna(0)
# df['Customer Happiness'] = df['Customer Happiness'].astype(int)
# Transform Categorical features from numeric to discrete
for i in cat_features:
    bins = df[i].quantile(np.linspace(0, 1, len(cat_variables[i]["values"]) + 1))

    df[i].replace(0, 0.1, inplace=True)
    df[i].replace(1, 0.99, inplace=True)
    df[i] = pd.cut(df[i], bins, labels=list(bins)[1:])
    df[i] = np.round(df[i].astype(float), 3)
    sorted = np.sort(df[i].unique())
    # if i == 'Occupation':
    #   sorted = np.sort(df[i].unique())[::-1]
    print(sorted)
    df[i] = df[i].replace(
        dict(
            zip(sorted, np.linspace(0, 1, len(cat_variables[i]["values"]) + 1).tolist())
        )
    )

# export
df_raw = df.copy()
df = df.fillna(0)
df.to_csv("data/transformed_data_raw.csv")

# Transform Categorical features from numeric to discrete
for i in cat_features:
    i = "Occupation"
    #  bins= df[i].quantile(np.linspace(0,1,len(cat_variables[i]['values']) + 1))
    #  df[i] = pd.cut(df[i],bins, labels=cat_variables[i]['names'])
    val_list = df[i].value_counts().index.tolist()
    if i == "Occupation":
        val_list = df[i].value_counts().index.tolist()[::-1]

    dictionary = dict(zip(val_list, cat_variables[i]["names"]))
    df[i] = df[i].map(dictionary)

df.to_csv("data/transformed_data_for_viewing.csv")


pp = sns.pairplot(
    data=df,
    y_vars=["Mobile Traffic"],
    x_vars=["Occupation", "Family Type", "Family Interest", "House Type"],
)
for ax in pp.axes.flatten():
    ax.tick_params(rotation=90)

pp = sns.pairplot(
    data=df,
    y_vars=["Mobile Traffic"],
    x_vars=[
        "Number Of Computers",
        "Number Of Tvs",
        "Number Of Pets",
        "Number Of Residents",
        "Number Of Phones",
        "Average Age",
    ],
)


pp = sns.pairplot(
    data=df,
    y_vars=["Mobile Traffic"],
    x_vars=[
        "Customer Happiness",
        "Time Spend On YouTube [min]",
        "Time Spend On TikTok [min]",
        "Time Spend On Instagram [min]",
        "Size of home [m2]",
        "Distance To Nearest Tower [m]",
    ],
)
