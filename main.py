import json
from representation import *
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.distribution import MultiprocessingDistributor


class IOMEP:
    def __init__(self, data_dir):
        r = Representation(data_dir)
    def load(self):
        data = []
        for subject in self.dir.rglob('*.csv'):
            # Load csv
            df = pd.read_csv(subject, engine='c', header=None)
            self.length.append(len(df))

            # Store subjects
            self.data.append(df)

    def build_reps(self):
        reps = {'raw': [], 'norm': [], 'tsfresh': [], 'tsfresh_fs': [], 'labels': []}
        for index, sub in enumerate(self.data):
            reps['labels'].append(sub[sub.columns[-1]])
            reps['raw'].append(sub[sub.columns[:-1]])
            reps['norm'].append(normalize(sub[sub.columns[:-1]], norm='max'))
            df = self.__compute_fe(index)
            reps['tsfresh'].append(df)
            reps['tsfresh_fs'].append(index)
        self.reps = reps

    def __compute_fe(self, index):
        path = Path('fe.csv')
        if not path.is_file():
            pass #self.fe, self.fs = start_fe()
        else:
            self.fe = pd.read_csv(path, engine='c')
        return self.fe.loc[self.length[index]:self.length[index+1]]

    def __compute_fs(self, index):
        path = Path('fs.csv')
        if self.fs:
            return self.fs.loc[self.length[index]:self.length[index+1]]
        else:
            self.fs = pd.read_csv(path, engine='c')
            return self.fs.loc[self.length[index]:self.length[index + 1]]


if __name__ == '__main__':
    a = IOMEP('data/dcs')
    a.load()
    a.build_reps()
