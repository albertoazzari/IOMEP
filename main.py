from representation import *
from model import *
from sklearn.model_selection import cross_val_score


class IOMEP:
    def __init__(self, data_dir):
        self.reps = Representation(data_dir).get_representations()
        self.models = Model().get_models()

    def intra_patient(self):
        labels = self.reps[-1]
        for rep in self.reps[:-1]:
            for idx, pat in enumerate(rep):
                for mod in self.models:
                    print(cross_val_score(mod, pat, labels[idx], cv=5))


if __name__ == '__main__':
    a = IOMEP('data/dcs')
