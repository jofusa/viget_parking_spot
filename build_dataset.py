"""
This file expects two folders ./available-crop/ and ./taken-crop/
containing the cropped and resized images created by crop.py



"""



import Image
import numpy as np
import pandas as pd
import os
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report
from sklearn.externals import joblib
import random


STANDARD_SIZE = (100, 100)

def img_to_feature_vector(filename):
        """
        takes a filename and turns it into a numpy feature array
        """
        img = Image.open(filename)
        img = img.resize(STANDARD_SIZE)
        img = list(img.getdata())
        img = map(list, img)
        img = np.array(img)
        return img.reshape(-1)






labels = []
data = []

available = ['./available-crop/' + f for f in os.listdir('./available-crop/')]
taken = ['./taken-crop/' + f for f in os.listdir('./taken-crop/')]


#Change the sample amount to increase or decrease the training size
# I have purposely resampled the data set to keep class inbalance from becoming
# an issue
for image in random.sample(taken,10):
    labels.append('taken')
    img = img_to_feature_vector(image)
    data.append(img)

for image in random.sample(available,10):
    labels.append('available')
    img = img_to_feature_vector(image)
    data.append(img)

data = np.array(data)

joblib.dump(data, 'data.pkl', compress=9)
joblib.dump(labels, 'labels.pkl', compress=9)
