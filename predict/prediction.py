# k-means clustering
import pandas as pd
import numpy as np
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler

# opening the datafile
with open('/Users/Jorg/BeCode2/Churn-project/data/clean_BankChurners.csv') as f:
    # read the csv file
    df = pd.read_csv(f)

# choosing the features (X)
X = df[['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 
       'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 
       'Income_Category', 'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 
       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
	    'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']]


# define the model
model = DBSCAN(eps=0.30, min_samples=3)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)

pca = PCA(n_components=3)
pca.fit(X)
PCA(n_components=2)
print('PCA explained variance ratio:', pca.explained_variance_ratio_)
print('PCA singular values:', pca.singular_values_)


import plotly.express as px

features = X

fig = px.scatter_matrix(df,
    dimensions=features,
    color="CLIENTNUM")
fig.update_traces(diagonal_visible=False)
fig.show()

pca = PCA()
components = pca.fit_transform(df)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

"""
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(df[row_ix, 0], df[row_ix, 1])
# show the plot
pyplot.show()
"""