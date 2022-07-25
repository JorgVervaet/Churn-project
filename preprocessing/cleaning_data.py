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

# replacing values
df['Attrition_Flag'].replace(['Attrited Customer', 'Existing Customer'],
                        [0, 1], inplace=True)
df['Gender'].replace(['M', 'F'],
                        [0, 1], inplace=True)
df['Marital_Status'].replace(['Unknown', 'Divorced', 'Single', 'Married'],
                        [0, 1, 2, 3], inplace=True)
df['Card_Category'].replace(['Blue', 'Silver', 'Gold', 'Platinum'],
                        [0, 1, 2, 3], inplace=True)
df['Education_Level'].replace(['Unknown', 'Uneducated', 'Education_Level', 'College', 'High School', 'Graduate', 'Post-Graduate', 'Doctorate'],
                        [0, 1, 2, 3, 4, 5, 6, 7], inplace=True)
df['Income_Category'].replace(['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'],
                        [0, 1, 2, 3, 4, 5], inplace=True)

# save the dataset to csv
df.to_csv('/Users/Jorg/BeCode2/Churn-project/data/clean_BankChurners.csv')