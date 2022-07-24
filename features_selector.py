
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
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


if __name__ == '__main__':
    data, labels = feat_sel()
    data = data.dropna(axis=1, inplace=False)
    labels = LabelEncoder().fit_transform(labels.to_numpy())

    # Create the RFE object and compute a cross-validated score.
    rf = RandomForestClassifier(n_jobs=-1)
    # The "accuracy" scoring shows the proportion of correct classifications

    min_features_to_select = 10  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=rf,
        step=1,
        cv=StratifiedKFold(5),
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
        n_jobs=-1
    )
    rfecv.fit(data, labels)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(
        range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
        rfecv.grid_scores_,
    )
    plt.show()
