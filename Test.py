import timeit

import numpy
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from emtiazi.KNN import KNeighboursClassifier
import matplotlib.pyplot as plt

dataset = pd.read_csv("arrhythmia_cleaned.csv")
classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X = sklearn.preprocessing.normalize(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

log = open('log.txt', 'w')

for m in ['manhattan', 'euclidean', 'cosine']:
    for n in [1, 7, 16, 30]:
        print('====================================================', file=log)
        print('metric: ', m, 'neighbors: ', n, file=log)
        start = timeit.default_timer()
        #
        knn = KNeighboursClassifier(n_neighbour=n, metric=m, classes=classes)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        #
        stop = timeit.default_timer()

        print('Time Of My Implementation: ', stop - start, file=log)
        print(y_pred, file=log)
        print(classification_report(y_test, y_pred), file=log)

        start = timeit.default_timer()
        #
        clf = KNeighborsClassifier(n_neighbors=n, metric=m)
        clf.fit(X_train, y_train)
        y_score = clf.predict(X_test)
        #
        stop = timeit.default_timer()

        print('Time Of Scikit Implementation: ', stop - start, file=log)
        print(y_score, file=log)
        print(classification_report(y_test, y_score), file=log)

        plt.figure()
        plt.figure(figsize=(12, 12))
        plt.scatter(numpy.linspace(0, len(y_test), len(y_test)), y_test, c='green', linestyle=':', linewidth=4)
        plt.plot(numpy.linspace(0, len(y_test), len(y_test)), y_test, c='green', linestyle=':', linewidth=4)
        plt.scatter(numpy.linspace(0, len(y_test), len(y_test)), y_pred, c='red', linewidth=2)
        plt.plot(numpy.linspace(0, len(y_test), len(y_test)), y_pred, c='red', linewidth=2)
        plt.scatter(numpy.linspace(0, len(y_test), len(y_test)), y_score, c='blue', linewidth=2)
        plt.plot(numpy.linspace(0, len(y_test), len(y_test)), y_score, c='blue', linewidth=2)
        plt.savefig('conclusion/' + str(n) + '_' + str(m) + '.png')
