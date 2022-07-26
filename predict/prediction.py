# k-means clustering
import pandas as pd
import numpy as np
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from kmodes.kmodes import KModes

# opening the datafile
with open('/Users/Jorg/BeCode2/Churn-project/data/clean_BankChurners.csv') as f:
    # read the csv file
    df = pd.read_csv(f)

# choosing the features (X)
X = df[['Customer_Age', 'Credit_Limit', 'Months_Inactive_12_mon', 'Total_Revolving_Bal', 
       'Contacts_Count_12_mon']]



pca = PCA(n_components=2)
pca.fit(X)
print('PCA explained variance ratio:', pca.explained_variance_ratio_)
print('PCA singular values:', pca.singular_values_)

import plotly.express as px

pca = PCA(n_components=2)
components = pca.fit_transform(X)

kmode = KModes(n_clusters=3, init = "Huang", n_init = 5, verbose=1, random_state=2)
kmode_res = kmode.fit_predict(X)


fig = px.scatter_matrix(
    components,
    #dimensions=components,
    color=kmode_res)

fig.update_traces(diagonal_visible=False)
fig.show()

filename = '/Users/Jorg/BeCode2/Churn-project/model/model_kmode.sav'
joblib.dump(kmode_res, filename)  
print(kmode_res.shape)
print(kmode_res)
print(components)
print(kmode.cluster_centroids_)
