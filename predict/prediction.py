# importing libraries
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA

# opening the datafile
with open('./data/clean_BankChurners_dummies.csv') as f:
    # read the csv file
    df = pd.read_csv(f)

# choosing the features (X)
X = df[['Credit_Limit', 'Total_Revolving_Bal', 'Avg_Utilization_Ratio',
       ]]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# use PCA to reduce the dimensions
pca = PCA(n_components=2)
pca.fit(X)
print('PCA explained variance ratio:', pca.explained_variance_ratio_)
print('PCA singular values:', pca.singular_values_)
components = pca.fit_transform(X)

# implement our model
kmeans = KMeans(n_clusters=3, n_init = 5, random_state=2)
kmeans_res = kmeans.fit_predict(X)

# produce a scatterplot
plt.scatter(components[:,0] , components[:,1], c=kmeans_res)
plt.show()

# save our model
filename = os.path.join('./model/model_KMeans.sav')
joblib.dump(kmeans, filename)  
