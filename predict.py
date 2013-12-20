
import Image
import numpy as np
import pandas as pd
import os
import sys
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.externals import joblib
import random

from optparse import OptionParser


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


def predict_parking_slot(image_path, show = False):
    img = img_to_feature_vector(image_path)
    estimator = joblib.load('./knn_model_crop_100_100.pkl')
    if show:
        Image.open(image_path).show()
    return estimator.predict(np.array(img))[0]

if __name__ == '__main__':


    parser = OptionParser("python ./predict.py IMG_FILE_TO_PREDICT")
    parser.add_option("-d", "--display",
                        action="store_true", dest="show", default=False,
                        help="open image that is predicted")

    (options, args) = parser.parse_args()



    print(predict_parking_slot(args[0], options.show))




