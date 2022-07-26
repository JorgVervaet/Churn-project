# k-means clustering
import pandas as pd
import numpy as np
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
from imblearn.over_sampling import RandomOverSampler

# opening the datafile
with open('/Users/Jorg/BeCode2/Churn-project/data/clean_BankChurners.csv') as f:
    # read the csv file
    df = pd.read_csv(f)

# choosing the features (X)
X = df[['Customer_Age', 
       'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 
       'Income_Category', 'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 
       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
	    'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']]


pca = PCA(n_components=2)
pca.fit(X)
print('PCA explained variance ratio:', pca.explained_variance_ratio_)
print('PCA singular values:', pca.singular_values_)

cost=[]
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
predict = clustering.fit_predict(X)
cost.append(predict)
    
plt.plot(predict, 'bx-')
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show()

import plotly.express as px


pca = PCA()
components = pca.fit_transform(X)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}


filename = '/Users/Jorg/BeCode2/Churn-project/model/model_dbscan.sav'
pickle.dump(cost, open(filename, 'wb'))