# Churn prediction project

## Scope

This API is made to take the input of a user and predict if that client is likely to churn or not.

## Workflow

First I made a program cleaning_data_dummies.py to get rid of columns we don't need and to make all data numerical. You can find this program in the preprocessing map.

Then we made a program to make our model. This program is called prediction.py and you can find this one in the predict folder.

I tested several models but for now KMeans gave the best results. I get a PCA explained variance ratio: [0.58962129 0.34704241]. So this is pretty good but I don't use that many features at the moment so this should be improved by implementing the use of the categorical data in the future.

## Usage

- Create a Python virtual environment (> 3.10) and activate it
- pip install -r requirements.txt
- python app.py

Or

- Install Docker (https://docs.docker.com/get-docker/)
- docker build . -t app  
- docker run -p 5001:5001 -t app

<img src="images/Screenshot 2.png" style="width:600px;height:200px;">

in both examples you will get something looking like this and you will be able to click on the url where the API will work.

Then you will get this screen:

<img src="images/Screenshot 4.png" style="width:600px;height:200px;">

Here you will have to fill in the Credit Limit, Total Revolving Balance and Average Utilization Ratio. After this you will get this screen to tell you in which cluster your client falls.

<img src="images/Screenshot 3.png" style="width:600px;height:200px;">


## Author

Jorg Vervaet


This is a graph of the clusters:

<img src="images/Screenshot 1.png" style="width:600px;height:200px;">