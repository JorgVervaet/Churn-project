# Churn prediction project

## Scope

This API is made to take the input of a user and predict if that client is likely to churn or not.

## Workflow

Get the n latest tweets of a given Twitter account by using the official Twitter API
Download all the images and videos shared in the tweets
Extract all the faces from the images and videos by using OpenCV cascade classifier
For the videos we focus on 1 pane out of 100 in order to lighten the process
We apply a DBScan clustering on the faces popping out and we keep one face per cluster
Encode the faces based on the face_recognition library
Sort the faces based on their similarity with the encoding of a reference image
Grey out the faces whose similarity is below a threshold of 0.9 (customisable)
Build a mosaic of the transformed faces
Usage

Create a Python virtual environment (> 3.9) and activate it
Create a Dev API account (V1.1) on Twitter and fill the credentials.json accordingly
pip install -r requirements.txt
python main.py
Example

python main.py

Please indicate a Twitter screen name to analyse:
> BarackObama

Please indicate an URL to the reference face:
> https://pbs.twimg.com/profile_images/1329647526807543809/2SGvnHYV_400x400.jpg

Please insert a similarity threshold (suggested: 0.9):
> 0.9 

How many tweets do you want to analyse (max = 3200)?
> 200

Getting the latest 200 tweets from the user...
200 tweets found

Getting images and videos URLS...
68 images and 31 videos found

Extracting faces from images...
177 potential faces found on images

Extracting faces from videos...
78 potential faces found on videos

Computing ratios and generating mosaic...

-------------------------------

The reference face appears 37 times out of 99 media shared by BarackObama (Ratio: 37.37%)
The reference face appears 37 times out of 255 valid faces popping up in the media shared by BarackObama (Ratio: 14.51%)

The mosaic has been generated : examples/BarackObama.png
alt text

Author

Louis de Viron - DataText SRL

Credentials

This tool is mainly based on the following python libraries:

opencv-python
face_recognition