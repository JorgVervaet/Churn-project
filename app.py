# prediction function
import joblib
import numpy as np
from flask import Flask, request, render_template

app=Flask(__name__, template_folder='templates')
@app.route('/')
def index():
    return render_template('index.html')


def ValuePredictor(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1, 5)
	loaded_model = joblib.load(open("/Users/Jorg/BeCode2/Churn-project/model/model_kmode.sav", "rb"))
	result = loaded_model.predict(to_predict)
	return result[0]

@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
		to_predict_list = request.form.to_dict()
		to_predict_list = list(to_predict_list.values())
		to_predict_list = list(map(int, to_predict_list))
		result = ValuePredictor(to_predict_list)	
		if int(result)== 1:
			prediction ='Income more than 50K'
		else:
			prediction ='Income less that 50K'		
		return render_template("result.html", prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)

