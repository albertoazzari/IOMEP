from model import *
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, cross_val_score, StratifiedKFold
import time


def intra_patient(
        reps: pd.DataFrame,
        labels: np.ndarray,
        models: list,
        configuration: str
) -> np.ndarray:
    scores = np.empty((len(reps.columns), len(reps.index), len(models)))
    for i, rep in enumerate(reps.columns):
        start_time = time.time()
        for (j, (x, y)) in enumerate(zip(reps[rep].to_numpy(), labels)):
            for (name, (k, model)) in zip(clf_names, enumerate(models)):
                n_samples, n_features = x.shape
                if name == "LIN SVM":  # Prefer dual=False when n_samples > n_features.
                    model.set_params(**{'dual': n_samples < n_features})
                print(f"\t rep {i}, patient {j}, model {name}")
                scores[i, j, k] = cross_val_score(model, np.nan_to_num(x), y, scoring='balanced_accuracy',
                                                  cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10),
                                                  n_jobs=-1).mean()
        print(f"Computing {rep} take - {(time.time() - start_time)} seconds")
    return scores


def inter_patient(
        reps: pd.DataFrame,
        labels: np.ndarray,
        models: list,
        n_classes: int = 2
) -> np.ndarray:
    scores = np.empty((len(reps.columns), len(models)))
    classes, frequency = np.unique(np.hstack(labels), return_counts=True)
    classes = classes[np.argsort(frequency)[-n_classes:]]
    y = np.hstack(labels)
    mask = np.isin(y, classes)
    y = y[mask]
    groups = np.hstack([np.ones(len(g)) * j for j, g in enumerate(labels)])[mask]
    for i, rep in enumerate(reps.columns):
        start_time = time.time()
        x = np.vstack(reps[rep])[mask, :]
        for (name, (j, model)) in zip(clf_names, enumerate(models)):
            n_samples, n_features = x.shape
            if name == "LIN SVM":  # Prefer dual=False when n_samples > n_features.
                model.set_params(**{'dual': n_samples < n_features})
            print(f"\t rep {i}, model {name}")
            scores[i, j] = cross_val_score(model, np.nan_to_num(x), y, scoring='balanced_accuracy',
                                           cv=StratifiedGroupKFold(n_splits=len(reps.columns)),
                                           groups=groups, n_jobs=-1).mean()
        print(f"Computing {rep} take - {(time.time() - start_time)} seconds")
    return scores


def inter_procedures(
        reps_dcs: pd.DataFrame,
        reps_tcs: pd.DataFrame,
        labels_dcs: np.ndarray,
        labels_tcs: np.ndarray,
        models: list
) -> np.ndarray:
    scores = np.empty((len(reps_dcs.columns), len(models)))
    classes = np.intersect1d(np.unique(np.hstack(labels_dcs), return_counts=True),
                             np.unique(np.hstack(labels_tcs), return_counts=True))
    mask_dcs = np.isin(np.hstack(labels_dcs), classes)
    mask_tcs = np.isin(np.hstack(labels_tcs), classes)
    y = np.hstack([np.hstack(labels_dcs)[mask_dcs], np.hstack(labels_tcs)[mask_tcs]])
    for (i, (dcs, tcs)) in enumerate(zip(reps_dcs.columns, reps_tcs.columns)):
        start_time = time.time()
        x = np.vstack([np.vstack(reps_dcs[dcs])[mask_dcs, :], np.vstack(reps_tcs[tcs])[mask_tcs, :]])
        for (name, (j, model)) in zip(clf_names, enumerate(models)):
            if name == "LIN SVM":
                n_samples, n_features = x.shape  # Prefer dual=False when n_samples > n_features.
                model.set_params(**{'dual': n_samples < n_features, 'max_iter': 1000, 'tol': 1e-5})
            scores[i, j] = cross_val_score(model, np.nan_to_num(x), y, scoring='balanced_accuracy',
                                           cv=StratifiedKFold(n_splits=5), n_jobs=-1).mean()
            print(f"rep:{i} with model:{j}")
        print(f"Computing {dcs} take - {(time.time() - start_time)} seconds")
    return scores


if __name__ == '__main__':
    opt = True
    if opt:
        clf_names = ["RF"]
        for configuration in ['', '_nolat']:
            clf = [RandomForestClassifier(max_samples=None, max_features='sqrt',
                                          n_estimators=3366 if configuration == '' else 644)]
            reps = pd.read_pickle(f'representations{configuration}.pickle')
            print(f'COMPUTING INTER-PROCEDURES - {"WS" if configuration == "" else "NL"}')
            reps_dcs = pd.DataFrame(reps['dcs'])
            reps_tcs = pd.DataFrame(reps['tcs'])
            np.save(f'data/results/inter_procedures{configuration}_opt.npy', inter_procedures(
                reps_dcs.loc[:,
                reps_dcs.columns != 'labels' if configuration == '' else ['mep', 'tsfresh', 'tsfresh_fs']],
                reps_tcs.loc[:,
                reps_tcs.columns != 'labels' if configuration == '' else ['mep', 'tsfresh', 'tsfresh_fs']],
                reps_dcs['labels'].to_numpy(),
                reps_tcs['labels'].to_numpy(),
                clf))
    else:
        clf_names, clf = Model().get_models()
        for configuration in ['', '_nolat']:
            reps = pd.read_pickle(f'representations{configuration}.pickle')
            for proc in reps.keys():
                # intra patient recognition
                print(f'COMPUTING INTRA PATIENT RECOGNITION - {"WS" if configuration == "" else "NL"}')
                rep = pd.DataFrame(reps[proc])
                np.save(f'data/results/intra_patient_{proc}{configuration}.npy', intra_patient(
                    rep.loc[:, rep.columns != 'labels' if configuration == '' else ['mep', 'tsfresh', 'tsfresh_fs']],
                    rep['labels'].to_numpy(),
                    clf,
                    configuration))

                # inter patient recognition task 2.1
                print(f'COMPUTING INTER PATIENT RECOGNITION - TASK 2_1 - {"WS" if configuration == "" else "NL"}')
                np.save(f'data/results/inter_patient2_{proc}{configuration}.npy', inter_patient(
                    rep.loc[:, rep.columns != 'labels' if configuration == '' else ['mep', 'tsfresh', 'tsfresh_fs']],
                    rep['labels'].to_numpy(),
                    clf,
                    2))

                # inter patient recognition 2.2
                print(f'COMPUTING INTER PATIENT RECOGNITION - TASK 2_2 - {"WS" if configuration == "" else "NL"}')
                np.save(f'data/results/inter_patient4_{proc}{configuration}.npy', inter_patient(
                    rep.loc[:, rep.columns != 'labels' if configuration == '' else ['mep', 'tsfresh', 'tsfresh_fs']],
                    rep['labels'].to_numpy(),
                    clf,
                    4))

            # inter procedures  recognition
            print(f'COMPUTING INTER-PROCEDURES - {"WS" if configuration == "" else "NL"}')
            reps_dcs = pd.DataFrame(reps['dcs'])
            reps_tcs = pd.DataFrame(reps['tcs'])
            np.save(f'data/results/inter_procedures{configuration}.npy', inter_procedures(
                reps_dcs.loc[:,
                reps_dcs.columns != 'labels' if configuration == '' else ['mep', 'tsfresh', 'tsfresh_fs']],
                reps_tcs.loc[:,
                reps_tcs.columns != 'labels' if configuration == '' else ['mep', 'tsfresh', 'tsfresh_fs']],
                reps_dcs['labels'].to_numpy(),
                reps_tcs['labels'].to_numpy(),
                clf))
