from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import statistics
from statistics import mode

def euc(x, y) :
    return distance.euclidean(x, y)

class ScrappyKNN() :
    def fit(self, features_train, labels_train) :
        self.features_train = features_train
        self.labels_train = labels_train

    def choose(self, labels) :
        return(mode(labels))

    def closest(self, item, K) :
        dist_min = {}
        for i in range(len(self.features_train)) :
            dist_min[i] = euc(item, self.features_train[i])
        dist_asc = sorted(dist_min.items(), key = lambda kv:(kv[1], kv[0]))
        dist_min = dist_asc[:K]
        labels = [self.labels_train[dist_min[i][0]] for i in range(len(dist_min))]
        return(self.choose(labels))

    def predict(self, features_test, K) :
        predictions = []
        for item in features_test:
            label = self.closest(item, K)
            predictions.append(label)
        return(predictions)



iris = datasets.load_iris()
features = iris['data']
labels = iris['target']

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .5)

my_classifier = ScrappyKNN()
my_classifier.fit(features_train, labels_train)

predictions = my_classifier.predict(features_test, 5)
print(accuracy_score(labels_test, predictions))
