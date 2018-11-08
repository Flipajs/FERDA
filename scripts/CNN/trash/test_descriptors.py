from __future__ import print_function
import h5py
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

if __name__ == '__main__':
    DIR = '/home/threedoid/cnn_descriptor'
    if len(sys.argv) > 1:
        WD = DIR + '/' + sys.argv[1]
    else:
        WD = DIR + '/data_cam3'

    with h5py.File(WD + '/results.h5', 'r') as hf:
        X = hf['data'][:]

    with h5py.File(WD + '/ids.h5', 'r') as hf:
        ids = hf['data'][:]

    with h5py.File(WD + '/classes.h5', 'r') as hf:
        y = hf['data'][:]

    # worst performance...
    # X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

    for i in range(6):
        print(np.sum(y_train==0))

    print("train ", X_train.shape)
    print("test ", X_test.shape)

    classifiers = {
        '3NeighborsClassifier': KNeighborsClassifier(3),
        'SVC linear C=0.025': SVC(kernel="linear", C=0.025),
        'SVC gamma=2, C=1': SVC(gamma=2, C=1),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier (n = 1000)': RandomForestClassifier(n_estimators=1000),
        'MLPClassifier': MLPClassifier(alpha=1),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'GaussianNB': GaussianNB(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis()}

    print("###############")
    import time
    for name, classifier in classifiers.iteritems():
        t = time.time()
        print(name)
        classifier.fit(X_train, y_train)
        training_t = time.time() - t

        t = time.time()
        y_pred = classifier.predict(X_test)
        predict_t = time.time() - t

        print("#correct/total: {}/{}, accuracy: {:.3%}".format(np.sum(y_test == y_pred),
                                                               X_test.shape[0],
                                                               accuracy_score(y_test, y_pred)))
        print("training t: {:.2f}s predict t: {:.2f}s".format(training_t, predict_t))
        print()
