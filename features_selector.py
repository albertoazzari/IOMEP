import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot
from numpy import mean, std
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


def feat_sel():
    data = []
    labels = []
    for fe in Path('data/dcs/fe').glob('*.csv'):
        # Load csv
        df = pd.read_csv(fe, engine='c', header=0, index_col=0)

        # Store subjects
        data.append(df)

    for sub in Path('data/dcs').glob('*.csv'):
        # Load csv
        df = pd.read_csv(sub, engine='c', header=None)

        # Store subjects
        labels.append(df[df.columns[-1]])
    return pd.concat(data, axis=0), pd.concat(labels)


# get a list of models to evaluate
def get_models():
    models = dict()
    for i in np.arange(start=10, stop=100, step=10):
        rfe = RFE(estimator=RandomForestClassifier(n_jobs=-1), n_features_to_select=i)
        model = RandomForestClassifier(n_jobs=-1)
        models[str(i)] = Pipeline(steps=[('s', rfe), ('m', model)])
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


if __name__ == '__main__':
    data, labels = feat_sel()
    data = data.dropna(axis=1, inplace=False)
    labels = LabelEncoder().fit_transform(labels.to_numpy())

    models = get_models()

    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, data, labels)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    # plot model performance for comparison
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()
