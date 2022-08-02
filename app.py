# import the libraries
import joblib
import numpy as np
from flask import Flask, request, render_template

app=Flask(__name__, template_folder='templates')
@app.route('/')
def index():
    return render_template('index.html')


def ValuePredictor(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1, 3)
	loaded_model = joblib.load('./model/model_KMeans.sav')
	result = loaded_model.predict(to_predict)
	return result[0]

@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
		to_predict_list = request.form.to_dict()
		to_predict_list = list(to_predict_list.values())
		to_predict_list = list(map(float, to_predict_list))
		result = ValuePredictor(to_predict_list)	

	if float(result)==0:
		prediction='cluster 0'		
	elif float(result)==1:
		prediction='cluster 1'
	elif float(result)==2:
		prediction='cluster 2'
	return render_template("result.html", prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)

