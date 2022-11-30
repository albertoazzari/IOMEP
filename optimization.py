import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from skopt import BayesSearchCV

# Creating the hyperparameter grid
param_dist = {"n_estimators": np.linspace(100, 5000, 10, dtype=int),
              "max_features": ["sqrt", "log2", None],
              "max_samples": [0.33, 0.66, None]}

data = pd.read_pickle('representations_nolat.pickle')
classes = np.intersect1d(np.unique(np.hstack(data['dcs']['labels']), return_counts=True),
                             np.unique(np.hstack(data['tcs']['labels']), return_counts=True))
mask_dcs = np.isin(np.hstack(data['dcs']['labels']), classes)
mask_tcs = np.isin(np.hstack(data['tcs']['labels']), classes)
y = np.hstack([np.hstack(data['dcs']['labels'])[mask_dcs], np.hstack(data['tcs']['labels'])[mask_tcs]])
x = np.vstack([np.vstack(data['dcs']['tsfresh'])[mask_dcs, :], np.vstack(data['tcs']['tsfresh'])[mask_tcs, :]])
rf = RandomForestClassifier()
# Instantiating RandomizedSearchCV object
tree_cv = RandomizedSearchCV(rf, param_dist, cv=5, n_jobs=-1)

tree_cv.fit(x, y)

# Print the tuned parameters and score
print("Tuned Random Forest Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
