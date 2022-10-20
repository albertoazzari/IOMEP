from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

clf_names = [
             "NN",
             "KNN",
             "LIN SVM",
             "RBF SVM",
             "RF"
             ]


class Model:
    def __init__(self):
        self.classifiers = [
            KNeighborsClassifier(n_neighbors=1, metric='euclidean'),
            KNeighborsClassifier(n_neighbors=10, metric='euclidean'),
            LinearSVC(),
            SVC(),
            RandomForestClassifier(oob_score=True)
        ]

    def get_models(self):
        return clf_names, self.classifiers
