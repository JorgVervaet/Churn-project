from fileinput import filename


import joblib


filename = "/Users/Jorg/BeCode2/Churn-project/model/model_kmode.sav"
loaded_model = joblib.load(open(filename, 'rb'))

print(loaded_model)