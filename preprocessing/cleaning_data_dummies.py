import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# opening the datafile
with open('/Users/Jorg/BeCode2/Churn-project/data/BankChurners.csv') as f:
    # read the csv file
    df = pd.read_csv(f)

# drop the unnecessary columns
df = df.drop(columns=['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])

for column_name in df.columns:
    column = df[column_name]
    # Get the count of Zeros in column 
    count = (column == 0).sum()
    print('Count of zeros in column ', column_name, ' is : ', count)


# Converting the categorical data of (Company_category and province) into numerical 
df_category = pd.get_dummies(df['Attrition_Flag'], prefix='Attrition_Flag')
df = pd.concat([df, df_category], axis=1)
df_category = pd.get_dummies(df['Gender'], prefix='Gender')
df = pd.concat([df, df_category], axis=1)
df_category = pd.get_dummies(df['Marital_Status'], prefix='Marital_Status ')
df = pd.concat([df, df_category], axis=1)
df_category = pd.get_dummies(df['Card_Category'], prefix='Card_Category ')
df = pd.concat([df, df_category], axis=1)
df_category = pd.get_dummies(df['Education_Level'], prefix='Education_Level ')
df = pd.concat([df, df_category], axis=1)
df_category = pd.get_dummies(df['Income_Category'], prefix='Income_Category ')
df = pd.concat([df, df_category], axis=1)

df = df.drop(columns=['Attrition_Flag', 'Gender', 'Marital_Status', 'Card_Category', 'Education_Level', 'Income_Category'])

# save the dataset to csv
df.to_csv('/Users/Jorg/BeCode2/Churn-project/data/clean_BankChurners_dummies.csv')