import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import time

from sklearn.model_selection import train_test_split
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_selection.selection import select_features
from tsfresh.utilities.dataframe_functions import impute


class Representation:
    def __init__(self, remove_lat: bool = False):
        # Path to reach the data directories
        self._dirs = [Path('data/dcs'), Path('data/tcs')]

        self._start_time = None

        dcs = {'raw': [], 'norm': [], 'mep': [], 'tsfresh': [], 'tsfresh_fs': [], 'labels': []}
        tcs = {'raw': [], 'norm': [], 'mep': [], 'tsfresh': [], 'tsfresh_fs': [], 'labels': []}
        self.procedures = {'dcs': dcs, 'tcs': tcs}

        # Encoding of the muscles acronyms into digits
        # (Exception for g and lsg which are encoded with different names but are the same muscle)
        self._encoding = {b'ah': 0, b'adm': 1, b'apb': 2, b'bb': 3, b'edcb': 4, b'ehl': 5, b'g': 6, b'lsg': 6, b'i': 7,
                          b'qf': 8, b'ra': 9, b'ta': 10, b'tb': 11}

        # Remove latency flag
        self._remove_lat = remove_lat

        # Core class function
        self._compute_representations()

        # Save self.procedures as dict in a file
        if self._remove_lat:
            _file_name = 'representations_nolat.pickle'
        else:
            _file_name = 'representations.pickle'
        with open(_file_name, 'wb') as f:
            pickle.dump(self.procedures, f)

    def _compute_representations(self, min_class_frequence: int = 5):
        print('Start representations building')
        self._start_time = time.time()
        # Loop through 2 directories dcs and tcs
        for dir in self._dirs:
            print(f'\tLoading and computing {dir} patients')
            # Read each patient file
            for obj in dir.rglob('*.csv'):
                # Load csv
                m = np.loadtxt(obj, delimiter=',', converters={-1: lambda s: self._encoding[s]})

                # Remove class which less frequent than min_class_frequence
                m = self.remove_infrequent(m, min_class_frequence)

                # Create list which contains the labelled muscles
                self.procedures[dir.stem]['labels'].append(m[:, -1])

                # Remove labels (which are stored in the last column) from signal
                m = m[:, :-1]

                # If the flag is true you want to compute the representations removing the latency
                if self._remove_lat:
                    m = self.remove_latency(m)

                # Create raw representation simpling adding the data as it is
                self.procedures[dir.stem]['raw'].append(m)

                # Create norm representation simpling adding the data scaled
                # scaled_values = ((val - min)*(new_max - new_min)/(max - min)) + new_min
                self.procedures[dir.stem]['norm'].append(self.compute_norm(m))

                # Create MEP representation extracting features from
                # [1] Stålberg, Erik et al., Standards for quantification of EMG and neurography
                self.procedures[dir.stem]['mep'].append(self.compute_mep(m))

        # Compute extraction of the features starting from raw representation
        # Select also the most discriminative 10 features
        self._compute_extraction_and_selection()
        print(f'All representations of both procedures computed in {(time.time() - self._start_time) / 3600:.2f} hours')

    @staticmethod
    def remove_latency(m):
        samples_nolat = []
        for i, sample in enumerate(m):
            peaks, _ = find_peaks(sample, height=(None, None), prominence=(5e-3, None), width=(None, None))
            # Note that not all signal have the 4 initial peaks
            try:
                if peaks[4] > 1000 or len(peaks) < 10:
                    first_peak = 0
                else:
                    first_peak = 4
            except IndexError:
                first_peak = 0

            samples_nolat.append(sample[peaks[first_peak]: peaks[-1]])
        return np.array(samples_nolat, dtype=object)

    @staticmethod
    def compute_norm(m):
        norm_samples = []
        for sample in m:
            norm_samples.append(((sample - np.min(sample)) * 2 / (np.max(sample) - np.min(sample)) - 1))
        return np.array(norm_samples, dtype=object)

    def _compute_fs(self, tsfresh, ldcs, ltcs):
        print('\tStart feature selection')
        start_time = time.time()
        # Compute the first feature selection using tsfresh and then re-select features using RFE
        y = np.hstack(
            [np.hstack(self.procedures['dcs']['labels']), np.hstack(self.procedures['tcs']['labels'])])
        fs = select_features(tsfresh, y, n_jobs=4, ml_task='classification', multiclass=True)
        rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10, step=10)
        _ = rfe.fit(fs, y)

        print(f'\tSelected {len(rfe.support_)} features in {(time.time() - start_time):.2f} seconds')
        self.procedures['dcs']['tsfresh_fs'] = [fs.loc[ldcs[i]:ldcs[i + 1]-1, rfe.support_]
                                                for i in range(len(ldcs) - 1)]
        self.procedures['tcs']['tsfresh_fs'] = [fs.loc[ltcs[i]:ltcs[i + 1]-1, rfe.support_]
                                                for i in range(len(ltcs) - 1)]

    @staticmethod
    def compute_mep(m):
        mep = np.empty((m.shape[0], 8))
        for i, sample in enumerate(m):
            peaks, info = find_peaks(sample, height=(None, None), prominence=(5e-3, None), width=(None, None))
            # Note that not all signal have the 4 initial peaks
            try:
                p = peaks[4]
                first_peak = 4
            except IndexError:
                first_peak = 0
            # Amplitude: Max amplitude
            feat1 = np.max(info['peak_heights'])

            # Area: Total area within duration
            feat2 = np.sum(info['widths'][first_peak:])

            # Duration: length of the signal response
            feat3 = peaks[-1] - peaks[first_peak]

            # Thickness: Area / Amplitude
            feat4 = feat3 / feat1

            # Size index: Normalized thickness, sometimes the value of amplitude is too low that the log10 is asyntotic
            try:
                feat5 = 2 * np.log10(feat1) + feat4
            except RuntimeWarning:
                feat5 = feat4

            # No. of phases: zero-crossing + 1
            zc = sample[peaks[first_peak]: peaks[-1]]
            feat6 = ((zc[:-1] * zc[1:]) < 0).sum() + 1

            # No. of turns: number of direction changes
            feat7 = len(peaks)

            # No.of spikes: number of peaks within the signal response
            feat8 = len(peaks[first_peak:-1])

            mep[i, :] = np.array([feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8])
        return mep

    def _compute_extraction_and_selection(self):
        print('\tStart feature extraction')
        # Build the dataframe in a form accepted by tsfresh
        data = self._build_stacked_df()

        # Compute the feature extraction of *data* and imputation of *ef*
        start_time = time.time()
        ef = impute(extract_features(timeseries_container=data, default_fc_parameters=ComprehensiveFCParameters(),
                                     column_id='id', column_value='value', n_jobs=4))
        print(f'\t{ef.shape[1]} features extract on {ef.shape[0]} signals in {(time.time() - start_time):.2f} seconds')

        # Store the length of the subjects both for dcs and tcs in order to reconstruct the patients list
        ldcs = [0]
        ldcs.extend([len(x) for x in self.procedures['dcs']['raw']])
        ldcs = np.cumsum(ldcs)

        ltcs = [ldcs[-1]]
        ltcs.extend([len(x) for x in self.procedures['tcs']['raw']])
        ltcs = np.cumsum(ltcs)

        self.procedures['dcs']['tsfresh'] = [ef.loc[ldcs[i]:ldcs[i + 1]-1, :] for i in range(len(ldcs) - 1)]
        self.procedures['tcs']['tsfresh'] = [ef.loc[ltcs[i]:ltcs[i + 1]-1, :] for i in range(len(ltcs) - 1)]

        # Compute the feature selection
        self._compute_fs(ef, ldcs, ltcs)

    def _build_stacked_df(self):
        data = np.vstack([np.vstack(
            [np.vstack([np.hstack([sample, np.full((3000 - len(sample)), np.nan)]) for sample in subject]) for subject
             in self.procedures['dcs']['raw']]), np.vstack(
            [np.vstack([np.hstack([sample, np.full((3000 - len(sample)), np.nan)]) for sample in subject]) for subject
             in self.procedures['tcs']['raw']])])

        stacked_data = []
        for i, ts in enumerate(data):
            x = [[x, i] for x in ts]
            stacked_data.append(x)
        df = pd.DataFrame(np.vstack(stacked_data), columns=['value', 'id'])
        return df.dropna(axis=0)

    @staticmethod
    def remove_infrequent(m: np.ndarray, min_class_frequence: int = 5):
        pat = pd.DataFrame(m)
        pat.columns = [*pat.columns[:-1], 'labels']
        pat = pat.groupby("labels").filter(lambda x: len(x) >= min_class_frequence)
        return pat.to_numpy()


if __name__ == '__main__':
    Representation()
    Representation(remove_lat=True)
