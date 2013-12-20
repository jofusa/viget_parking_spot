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


labels = joblib.load( 'labels.pkl')
data = joblib.load('data.pkl')

# PCA will reduce the dimentionality of the problem to something manageble
pca = RandomizedPCA(n_components=20)
knn = KNeighborsClassifier()
pipe = Pipeline(steps=[('pca', pca), ('knn', knn)])

# The components to try
n_components = [10, 20, 40, 64]

#Setup a pipline to first apply PCA -> then predict with
#The Grid Search will preform cross validation and choose the optimal
#number of principle components
estimator = GridSearchCV(pipe,
                        dict(pca__n_components=n_components
                            ))
estimator.fit(data, labels)

print(classification_report(labels,estimator.predict(data)))

# If you are trying this one, change the model name or you will overwrite the
# old model
joblib.dump(estimator, 'knn_model_crop_100_100.pkl', compress=9)

