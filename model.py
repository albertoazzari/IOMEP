from sklearn.neighbors import KNeighborsClassifier
from tslearn.metrics import dtw
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier


class Model:
    def __init__(self):
        self.classifiers = [
            KNeighborsClassifier(n_neighbors=1, metric='euclidean'),
            KNeighborsClassifier(n_neighbors=10, metric='euclidean'),
            LinearSVC(max_iter=1000, tol=1e-5),
            SVC(),
            RandomForestClassifier(oob_score=True)
        ]
        self.clf_names = [
            "NN",
            "KNN",
            "LIN SVM",
            "RBF SVM",
            "RF"
        ]

    def get_models(self):
        return self.clf_names, self.classifiers
