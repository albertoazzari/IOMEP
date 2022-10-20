import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from scipy.signal import find_peaks
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import impute


class Representation:
    def __init__(self):
        self.dirs = [Path('data/dcs'), Path('data/tcs')]
        self.representation = {'dcs': {'raw': [], 'norm': [], 'mep': [], 'tsfresh': [], 'tsfresh_fs': [], 'labels': []},
                               'tcs': {'raw': [], 'norm': [], 'mep': [], 'tsfresh': [], 'tsfresh_fs': [], 'labels': []}}
        self.encoding = {b'ah': 0, b'adm': 1, b'apb': 2, b'bb': 3, b'edcb': 4, b'ehl': 5, b'g': 6, b'lsg': 6, b'i': 7,
                         b'qf': 8, b'ra': 9, b'ta': 10, b'tb': 11}
        self.compute_representations()
        with open('representations.pickle', 'wb') as f:
            pickle.dump(self.representation, f)

    def compute_representations(self):
        print('Read patients data')
        for dir in self.dirs:
            for obj in dir.rglob('*.csv'):
                # Load csv
                m = np.loadtxt(obj, delimiter=',', converters={-1: lambda s: self.encoding[s]})
                m = self.remove_infrequent(m)
                self.representation[dir.stem]['raw'].append(m[:, :-1])
                self.representation[dir.stem]['norm'].append(normalize(m[:, :-1], norm='max'))
                self.representation[dir.stem]['mep'].append(self.compute_mep(m[:, :-1]))
                self.representation[dir.stem]['labels'].append(m[:, -1])
        self.compute_extraction()
        # self.representation[dir.stem]['tsfresh'].append(
            # self.representation[dir.stem]['tsfresh_fs'].append([x[self.compute_fs(self.representation)] for x in self.representation[dir.stem]['tsfresh']])

    @staticmethod
    def compute_fs(data):
        features = data['dcs']['tsfresh'][0].columns.to_numpy()
        x = np.vstack([np.vstack(data['dcs']['tsfresh']), np.vstack(data['tcs']['tsfresh'])])
        x = np.nan_to_num(x)
        y = np.hstack([np.hstack(data['dcs']['labels']), np.hstack(data['tcs']['labels'])])

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

        # Create decision tree classifer object
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        rf.fit(X_train, y_train)
        # the permutation based importance
        perm_importance = permutation_importance(rf, X_train, y_train, scoring='accuracy', n_repeats=5, n_jobs=-1)
        sorted_idx = perm_importance.importances_mean.argsort()
        features = features[sorted_idx]
        scores = []
        for n in np.arange(start=10, stop=len(features), step=10):
            rf = RandomForestClassifier(oob_score=True, n_jobs=-1).fit(X_train[:, sorted_idx[:n]], y_train)
            scores.append(rf.score(X_test[:, sorted_idx[:n]], y_test))
        # fig = plt.figure(figsize=(24, 18))
        # sns.set(font_scale=2.5)
        # sns.despine(bottom=True, left=True)
        df_scores = pd.DataFrame({"features employed": np.arange(start=10, stop=len(features), step=10), "accuracy": scores})
        # ax = sns.lineplot(data=df_scores, x='features employed', y="accuracy")
        # plt.title(f'Feature Selection Performances')
        # plt.savefig(f'data/plots/fs.png')
        # plt.show()
        n_feat = np.where(np.array(scores) > 0.9)[0][0]
        return features[:df_scores["features employed"][n_feat]]

    @staticmethod
    def compute_mep(data):
        meps = []
        for sig in data:
            feat = []
            fp_pos = find_peaks(sig, height=(None, None), prominence=(None, None), width=(None, None))
            fp_neg = find_peaks(-sig, height=(None, None), prominence=(None, None), width=(None, None))
            feat.append(np.max(fp_pos[1]['prominences']))
            feat.append(fp_pos[1]['widths'][np.argmax(fp_pos[1]['prominences'])])
            feat.append(np.sum(sig[
                               fp_pos[1]['left_bases'][np.argmax(fp_pos[1]['prominences'])]:fp_pos[1]['right_bases'][
                                   np.argmax(fp_pos[1]['prominences'])]]))
            feat.append(feat[2] / feat[0])
            feat.append(2 * np.log10(feat[0]) + feat[3])
            feat.append(((sig[:-1] * sig[1:]) < 0).sum())
            feat.append(len(fp_pos[0]) + len(fp_neg[0]))
            feat.append(len(fp_pos[0]))
            meps.append(feat)
        return np.array(meps)

    def compute_extraction(self):
        data = self.build_stacked_df(self.representation)
        data = impute(extract_features(timeseries_container=data, default_fc_parameters=ComprehensiveFCParameters(), column_id='id', column_value='value', column_kind='kind', n_jobs=4))
        with open('extr.pickle', 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def build_stacked_df(reps):
        data = np.vstack([np.vstack(reps['dcs']['raw']), np.vstack(reps['tcs']['raw'])])
        labels = np.hstack([np.hstack(reps['dcs']['labels']), np.hstack(reps['tcs']['labels'])])
        df = pd.DataFrame()
        for ((i, ts), label) in zip(enumerate(data), labels):
            x = [[x, i, label] for x in ts]
            df = df.append(x, ignore_index=True)
        df.columns = ['value', 'id', 'kind']
        return df

    @staticmethod
    def remove_infrequent(data: np.ndarray):
        pat = pd.DataFrame(data)
        pat.columns = [*pat.columns[:-1], 'labels']
        pat = pat.groupby("labels").filter(lambda x: len(x) >= 5)
        return pat.to_numpy()


if __name__ == '__main__':
    Representation()
    #data = pd.read_pickle('representations.pickle')
    #Representation.compute_fs(data)
