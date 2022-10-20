# importing pandas as pd
import numpy as np
import pandas as pd
from pathlib import Path


def convert(path):
    # Read and store content
    # of an Excel file

    path = Path(path)
    l = []
    for file in path.glob('*.xlsx'):
        read_file = pd.read_excel(file, index_col=0).transpose()
        read_file['labels'] = [''.join([i[0].lower() for i in l.split()[:-1]]) for l in read_file.index.tolist()]
        np.savetxt(file.with_suffix('.csv'), read_file.to_numpy(), fmt='%s', delimiter=',')


if __name__ == '__main__':
    convert('/Users/albertoazzari/Downloads/pazienti_iom_ML/tcs')
