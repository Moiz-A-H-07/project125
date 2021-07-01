import cv2
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

X=np.load('image.npz')['arr_0']
y=pd.read_csv("labels.csv")["labels"]
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=3500, test_size=500) 
#scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

def getprediction(image):
    impil=Image.open(image)
    imagebw=impil.covert('L')
    imageresized=imagebw.resize((28,28),Image.ANTIALIAS)
    pixelfilter=20
    minpixel=np.percentile(imageresized,pixelfilter)
    imageresizedinverted=np.clip(imageresized-minpixel,0,255)
    maxpixel=np.max(imageresized)
    imageresizedinverted=np.asarray(imageresizedinverted)/maxpixel
    testsample=np.array(imageresizedinverted).reshape(1784)
    testpredict=clf.predict(testsample)
    return testpredict[0]
