# %%
from sklearn import datasets
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

# LOADING DATASETS
digits = datasets.load_digits()
features = digits.data
target = digits.target

# CREATING SIMULATED DATASETS
# Regression
features, target, coefficients = make_regression(
    n_samples=100,
    n_targets=1,
    n_features=3,
    n_informative=3,
    noise=0.0,
    coef=True,
    random_state=1
)

print('Feature matrix\n', features[:3])
print('Target vector\n', target[:3])

# Classification
features, target = make_classification(
    n_samples=100,
    n_informative=3,
    n_features=3,
    n_classes=2,
    n_redundant=0,
    weights=[.25, .75],
    random_state=1
)

print('Feature matrix\n', features[:3])
print('Target vector\n', target[:3])

# Clustering
features, target = make_blobs(
    n_samples=100,
    n_features=2,
    centers=3,
    cluster_std=.5,
    shuffle=True,
    random_state=1
)

print('Feature matrix\n', features[:3])
print('Target vector\n', target[:3])

plt.scatter(features[:, 0], features[:, 1], c=target)
plt.show()

# LOADING A CSV FILE
# csv
url = 'https://raw.githubusercontent.com/gafnts/Inflation-forecasting/main/data/main.csv'
dataframe = pd.read_csv(url)
dataframe.head()

#json
url = 'https://api.exchangerate-api.com/v4/latest/USD'
dataframe = pd.read_json(url, orient = 'columns')
dataframe.head()
new_data = dataframe[['date', 'rates']]

# sql database
database_connection = create_engine('sqlite:///sample.db')
dataframe = pd.read_sql_query('SELECT * FROM data', database_connection)
