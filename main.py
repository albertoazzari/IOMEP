import json
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
        dir = Path(data_dir)
        if dir.is_dir():
            self.dir = dir
            self.data = []
        else:
            raise ValueError("input path not directory")

    def load(self):
        data = []
        for subject in self.dir.rglob('*.csv'):
            # Load csv
            df = pd.read_csv(subject, engine='c', header=None)

            # Store subjects
            self.data.append(df)

    def build_reps(self):
        reps = {'raw': [], 'norm': [], 'tsfresh': [], 'tsfresh_fs': [], 'labels': []}
        for index, sub in enumerate(self.data):
            reps['labels'].append(sub[sub.columns[-1]])
            reps['raw'].append(sub[sub.columns[:-1]])
            reps['norm'].append(normalize(sub[sub.columns[:-1]], norm='max'))
            df = self.__compute_fe(sub[sub.columns[:-1]], index)
            reps['tsfresh'].append(df)
            reps['tsfresh_fs'].append(self.__compute_fs(df))
        self.reps = reps

    def __compute_fe(self, subject, index):
        path = Path('pat' + str(index) + '.csv')
        if not path.is_file():
            self.dist = MultiprocessingDistributor(n_workers=4, disable_progressbar=False)
            t = pd.DataFrame(np.ravel(np.transpose(subject)), columns=['value'])
            t['kind'] = pd.Series(['x' for x in range(len(t.index))])
            t['id'] = pd.Series([int(x / 3000) for x in range(len(t.index))])
            df_fe = extract_features(timeseries_container=subject,
                                     column_id="id", column_kind="kind", column_value="value", distributor=self.dist)
            df_fe.to_csv(path)
        else:
            df_fe = pd.read_csv(path, engine='c')
        return df_fe.dropna(axis=1, inplace=False)

    def __compute_fs(self, subject):
        path = Path('fs.csv')
        if not path.is_file():
            data = pd.concat(self.data, axis=0)
            data.columns = [*data.columns[:-1], 'labels']
            X_train, X_test, y_train, y_test = train_test_split(data[data.columns[:-1]], data['labels'].to_numpy(), test_size=0.33, random_state=42)
            sel = SelectFromModel(RandomForestClassifier(n_estimators=1000))
            sel.fit(X_train, y_train)
            selected_feat = pd.Series(X_train.columns[(sel.get_support())])
            selected_feat.to_csv()
        else:
            selected_feat = pd.read_csv(path)
        return subject[selected_feat.to_numpy()]


if __name__ == '__main__':
    a = IOMEP('data/dcs')
    a.load()
    a.build_reps()
