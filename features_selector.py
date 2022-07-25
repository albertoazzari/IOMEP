import pandas as pd
from pathlib import Path
from tsfresh.feature_extraction import extract_features


def read_data(path):

    m = pd.DataFrame()
    for sub in Path(path).glob('*.csv'):
        # Load csv
        df = pd.read_csv(sub, engine='c', header=None)
        m = pd.concat([m, df], axis=0, ignore_index=True)

    m = m.to_numpy()
    df = pd.DataFrame()
    for i, row in enumerate(m):
        ts = pd.DataFrame([[x, i, row[-1]] for x in row[:-1]])
        df = pd.concat([df, ts], axis=0, ignore_index=True)
    df.columns = ['value', 'id', 'kind']

    return df


def feat_ext(data):
    print('starting feature extraction')
    df_fe = extract_features(timeseries_container=data,
                             column_id="id", column_kind="kind", column_value="value", n_jobs=4)
    df_fe.to_csv('fe.csv')


if __name__ == '__main__':
    data = read_data('data/dcs')
    feat_ext(data)
    print('ok')
