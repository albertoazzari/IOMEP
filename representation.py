from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import normalize
from tsfresh import extract_features
from tsfresh import extract_relevant_features

label = {b'ah': 0, b'apb': 1, b'bb': 2, b'edcb': 3, b'qf': 4, b'ta': 5}


class Representation:
    def __init__(self, path):
        dir = Path(path)
        if dir.is_dir():
            self.dir = dir
            self.nmeps = []
            self.representations = {'raw': [], 'norm': [], 'tsfresh': [], 'tsfresh_fs': [], 'labels': []}
            self.__read_data()
            self.__extract_features()
            self.__select_features()
            with open('reps.pickle', 'wb') as f:
                pickle.dump(self.representations, f)
        else:
            raise ValueError("input path not directory")

    def __read_data(self):
        for obj in self.dir.rglob('*.csv'):
            # Load csv
            m = np.loadtxt(obj, delimiter=',', converters={-1: lambda s: label[s]})
            self.representations['raw'].append(m[:, :-1])
            self.representations['norm'].append(normalize(m[:, :-1], norm='max'))
            self.representations['labels'].append(m[:, -1])

    def __extract_features(self):
        m = np.append(np.vstack(self.representations['raw']),
                      np.hstack(self.representations['labels']).reshape((-1, 1)), axis=1)
        d = []
        for i, row in enumerate(m):
            d.append([[x, i, row[-1]] for x in row[:-1]])
        feat_ext = extract_features(timeseries_container=pd.DataFrame(np.vstack(d), columns=['value', 'id', 'kind']),
                             column_id="id", column_kind="kind", column_value="value", n_jobs=4)
        self.nmeps.append(0)
        [self.nmeps.append(len(x)) for x in self.representations['labels']]
        self.nmeps = np.cumsum(self.nmeps)
        for idx in range(len(self.nmeps)-1):
            self.representations['tsfresh'].append(feat_ext.iloc[self.nmeps[idx]:self.nmeps[idx+1], :])

    def __select_features(self):
        m = np.append(np.vstack(self.representations['raw']),
                      np.hstack(self.representations['labels']).reshape((-1, 1)), axis=1)
        d = []
        for i, row in enumerate(m):
            d.append([[x, i, row[-1]] for x in row[:-1]])
        y = pd.Series(np.hstack(self.representations['labels']))
        feat_sel = extract_relevant_features(pd.DataFrame(np.vstack(d), columns=['value', 'id', 'kind']), y, column_id="id", column_kind="kind", column_value="value", n_jobs=4)
        for idx in range(len(self.nmeps)-1):
            self.representations['tsfresh_fs'].append(feat_sel.iloc[self.nmeps[idx]:self.nmeps[idx+1], :])

