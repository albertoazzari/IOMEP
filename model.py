from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class Model:
    def __init__(self):
        self.models = []
        self.models.append(KNeighborsClassifier(n_neighbors=1, metric='seuclidean'))
        self.models.append(KNeighborsClassifier(n_neighbors=10, metric='seuclidean'))
        self.models.append(SVC(kernel='linear'))
        self.models.append(SVC(kernel='rbf'))
        self.models.append(RandomForestClassifier())

    def get_models(self):
        return self.models
