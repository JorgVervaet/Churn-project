# k-means clustering
import pandas as pd
import numpy as np
from numpy import unique
from numpy import where
import hdbscan
from sklearn.datasets import make_classification
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler

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

scaler = StandardScaler()
X = scaler.fit_transform(X)


pca = PCA(n_components=2)
components = pca.fit_transform(X)

hdb = hdbscan.HDBSCAN()
hdb_res = hdb.fit_predict(X)

plt.scatter(components[:,0] , components[:,1], c=hdb_res)
plt.show()


filename = '/Users/Jorg/BeCode2/Churn-project/model/model_kmode.sav'
joblib.dump(hdb, filename)  
print(hdb_res)
print(components)
print(hdb_res.labels_.max())
