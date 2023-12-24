import numpy

class KNeighboursClassifier:
    """Classifier implementing the k-nearest neighbors vote.
    n_neighbors : int, optional (default = 5)
    metric : string or callable, default 'euclidean'
            {'manhattan', 'euclidean', 'cosine'}
    classes : an array
    """

    X_train = []
    y_train = []

    def __init__(self, n_neighbour=5, metric='euclidean', classes=[]):
        self.n_neighbour = n_neighbour
        self.metric = metric
        self.classes = classes

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        y_pred = []
        for i in range(len(x_test)):
            distances = []
            for j in range(len(self.X_train)):
                distances.append(self.__calculate_distance(self.X_train[j], x_test[i]))

            neighbour_index = numpy.zeros((self.n_neighbour), int)
            neighbour_min = numpy.zeros((self.n_neighbour), float)
            for k in range(len(neighbour_index)):
                min = distances[0]
                neighbour_min[k] = distances[0]
                neighbour_index[k] = 0
                for o in range(len(distances)):
                    if distances[o] < min and (o not in neighbour_index):
                        min = distances[o]
                        neighbour_min[k] = distances[o]
                        neighbour_index[k] = o

            neighbour_classes = []
            for k in range(self.n_neighbour):
                neighbour_classes.append(self.y_train[neighbour_index[k]])

            mode = numpy.zeros((len(self.classes)), int)
            max, max_index = 0, 0
            for item in neighbour_classes:
                item -= 1
                mode[int(item)] += 1
                if mode[int(item)] > max:
                    max = mode[int(item)]
                    max_index = int(item)

            y_pred.append(self.classes[max_index])

        return y_pred

    def __calculate_distance(self, x, y):
        distance = 0
        if self.metric == 'euclidean':
            for i in range(len(x)):
                distance += (x[i] - y[i]) ** 2
            distance = distance ** 0.5
        elif self.metric == 'manhattan':
            for i in range(len(x)):
                distance += abs(x[i] - y[i])
        elif self.metric == 'cosine':
            sumxx, sumxy, sumyy = 0, 0, 0
            for i in range(len(x)):
                sumxx += x[i] ** 2
                sumyy += y[i] ** 2
                sumxy += x[i] * y[i]
            distance = sumxy / ((sumxx ** 0.5) * (sumyy ** 0.5))
        return distance
