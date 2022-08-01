# k-means clustering
from matplotlib.cbook import ls_mapper
import pandas as pd
import seaborn as sns
from pyspark.sql import SparkSession
import numpy as np
from pyspark.mllib.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from matplotlib import pyplot
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA

# opening the datafile
"""
with open('/Users/Jorg/BeCode2/Churn-project/data/clean_BankChurners_dummies.csv') as f:
    # read the csv file
    df = pd.read_csv(f)
"""
spark = SparkSession.builder.appName("DataFrame").getOrCreate()

df = spark.read.csv('/Users/Jorg/BeCode2/Churn-project/data/clean_BankChurners_dummies.csv')

# choosing the features (X)
X = df[["Contacts_Count_12_mon","Credit_Limit","Total_Revolving_Bal","Avg_Utilization_Ratio","Gender_M","Gender_F"]]

X = spark.createDataFrame(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)


pca = PCA(n_components=2)
components = pca.fit_transform(X)
print('PCA explained variance ratio:', pca.explained_variance_ratio_)
print('PCA singular values:', pca.singular_values_)


kmeans = KMeans(k=2)
kmeans.setSeed(1)
kmeans.setWeightCol("weighCol")
kmeans.setMaxIter(10)
kmeans.getMaxIter()
kmeans.clear(kmeans.maxIter)
model = kmeans.fit(df)
model.getDistanceMeasure()

sns.scatterplot(x=components[:,0], y=components[:,1], s=50, c=kmeans_res, palette='tab10')
plt.show()

filename = '../model/model_kmode.sav'
joblib.dump(kmeans, filename)  
print(kmeans_res.shape)
print(kmeans_res)
print(components)
print(kmeans.cluster_centers_)
print(kmeans_res.shape)
