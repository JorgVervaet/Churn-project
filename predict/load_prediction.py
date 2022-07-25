from fileinput import filename


import pickle
from kmodes.kmodes import KModes


filename = "/Users/Jorg/BeCode2/Churn-project/model/model_kmode.sav"
loaded_model = pickle.load(open(filename, 'rb'))

print(loaded_model)