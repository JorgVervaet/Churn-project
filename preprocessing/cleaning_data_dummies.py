#importing the libraries
import pandas as pd

# opening the datafile
with open('/Users/Jorg/BeCode2/Churn-project/data/BankChurners.csv') as f:
    # read the csv file
    df = pd.read_csv(f)

# drop the unnecessary columns
df = df.drop(columns=['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])

# Converting the categorical data into numerical 
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

# drop the original categorical columns
df = df.drop(columns=['Attrition_Flag', 'Gender', 'Marital_Status', 'Card_Category', 'Education_Level', 'Income_Category'])

# save the dataset to csv
df.to_csv('./data/clean_BankChurners_dummies.csv')