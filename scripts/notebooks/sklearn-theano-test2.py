import numpy as np
import matplotlib.pyplot as plt
from sklearn_theano.datasets import load_sample_image
from sklearn_theano.feature_extraction import OverfeatLocalizer, OverfeatTransformer
from sklearn.mixture import GMM
import cv2
import time

def get_features(p, collision_chunks):
    features = {}
    i = 0
    for ch in p.chm.chunk_gen():
        if ch in self.collision_chunks:
            continue

        # if i > 20:
        #     break
        X = self.get_data(ch)

        i += 1
        features[ch.id()] = X

        print i

    return features

if __name__ == "__main__":
    # X = load_sample_image("cat_and_dog.jpg")
    X2 = cv2.imread('/Users/flipajs/Desktop/Screen Shot 2016-10-10 at 11.06.20.png')
    X3 = cv2.imread('/Users/flipajs/Desktop/Screen Shot 2016-10-10 at 11.06.25.png')

    oft = OverfeatTransformer()

    # f = oft.transform(X)
    t = time.time()
    f2 = oft.transform(np.array([X2, X3]))
    print time.time() - t
    t = time.time()
    f3 = oft.transform(X3)
    print time.time() - t

    print f2.shape
    print f3.shape

    print "test"