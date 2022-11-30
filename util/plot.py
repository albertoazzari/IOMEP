import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

from model import Model

clf_names, _ = Model().get_models()


def pie_chart():
    feats = pd.read_pickle('representations.pickle')['dcs']['tsfresh'][0].columns.to_list()
    k = [re.findall('__(.*)__|__(.*)', i) for i in feats]
    a = [x[1] if x[0] == '' else x[0] for x in [item for sublist in k for item in sublist]]
    mask = [x.__contains__('__') for x in a]
    c = [x[0:x.find('__', 0)] if x.find('__', 0) else x for x in np.asarray(a)[mask]]
    c.extend(np.asarray(a)[not mask])
    s = pd.Series(c).value_counts()
    s = s[s.values/np.sum(s.values) > 0.009]
    colors = sns.color_palette('pastel')
    fig = plt.figure(figsize=(24, 18))
    sns.set(font_scale=2.5)
    sns.despine(bottom=True, left=True)
    plt.title('Overview of the features extracted')
    plt.pie(s, labels=s.index.tolist(), autopct='%0.0f%%', colors=colors)
    plt.savefig(f'data/plots/feats_pie.png')
    plt.show()


def task1_sp(proc):
    sns.set(font_scale=2.5)
    figure, axes = plt.subplots(2, 1, sharex=True,
                                figsize=(24, 18))
    figure.suptitle(f'TASK 1 - {proc.upper()}')
    axes[0].set_title('Whole Signal')
    axes[1].set_title('No Latency')

    task_ws = np.load(f'data/results/intra_patient_{proc}.npy', allow_pickle=True)
    rep, sub, clf = task_ws.shape
    hm = np.zeros((rep, clf))
    for i in range(rep):
        for j in range(clf):
            hm[i, j] = task_ws[i, :, j].mean()
    task_ws = pd.DataFrame(hm,
                           index=['RAW', 'NORM', 'MEP', 'TSFRESH', 'TSFRESH_FS'],
                           columns=clf_names)
    task_nl = np.load(f'data/results/intra_patient_{proc}_nolat.npy', allow_pickle=True)
    rep, sub, clf = task_nl.shape
    hm = np.zeros((rep, clf))
    for i in range(rep):
        for j in range(clf):
            hm[i, j] = task_nl[i, :, j].mean()
    task_nl = pd.DataFrame(hm,
                           index=['MEP', 'TSFRESH', 'TSFRESH_FS'],
                           columns=clf_names)

    sns.heatmap(task_ws, annot=True, fmt=".4f", linewidths=.5, vmin=0, vmax=1, cmap='mako', ax=axes[0], cbar=False)
    sns.heatmap(task_nl, annot=True, fmt=".4f", linewidths=.5, vmin=0, vmax=1, cmap='mako', ax=axes[1], cbar=False)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30)

    plt.savefig(f'data/plots/task1_{proc}.png')
    plt.show()


def task2_1_sp(proc):
    sns.set(font_scale=2.5)
    figure, axes = plt.subplots(2, 1, sharex=True,
                                figsize=(24, 18))
    figure.suptitle(f'TASK 2_1 - {proc.upper()}')
    axes[0].set_title('Whole Signal')
    axes[1].set_title('No Latency')

    task_ws = pd.DataFrame(np.load(f'data/results/inter_patient2_{proc}.npy', allow_pickle=True),
                           index=['RAW', 'NORM', 'MEP', 'TSFRESH', 'TSFRESH_FS'],
                           columns=clf_names)
    task_nl = pd.DataFrame(np.load(f'data/results/inter_patient2_{proc}_nolat.npy', allow_pickle=True),
                           index=['MEP', 'TSFRESH', 'TSFRESH_FS'],
                           columns=clf_names)

    sns.heatmap(task_ws, annot=True, fmt=".4f", linewidths=.5, vmin=0, vmax=1, cmap='mako', ax=axes[0], cbar=False)
    sns.heatmap(task_nl, annot=True, fmt=".4f", linewidths=.5, vmin=0, vmax=1, cmap='mako', ax=axes[1], cbar=False)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30)

    plt.savefig(f'data/plots/task2_1_{proc}.png')
    plt.show()


def task2_2_sp(proc):
    sns.set(font_scale=2.5)
    figure, axes = plt.subplots(2, 1, sharex=True,
                                figsize=(24, 18))
    figure.suptitle(f'TASK 2_2 - {proc.upper()}')
    axes[0].set_title('Whole Signal')
    axes[1].set_title('No Latency')

    task_ws = pd.DataFrame(np.load(f'data/results/inter_patient4_{proc}.npy', allow_pickle=True),
                           index=['RAW', 'NORM', 'MEP', 'TSFRESH', 'TSFRESH_FS'],
                           columns=clf_names)
    task_nl = pd.DataFrame(np.load(f'data/results/inter_patient4_{proc}_nolat.npy', allow_pickle=True),
                           index=['MEP', 'TSFRESH', 'TSFRESH_FS'],
                           columns=clf_names)

    sns.heatmap(task_ws, annot=True, fmt=".4f", linewidths=.5, vmin=0, vmax=1, cmap='mako', ax=axes[0], cbar=False)
    sns.heatmap(task_nl, annot=True, fmt=".4f", linewidths=.5, vmin=0, vmax=1, cmap='mako', ax=axes[1], cbar=False)

    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30)

    plt.savefig(f'data/plots/task2_2_{proc}.png')
    plt.show()


def task3():
    sns.set(font_scale=2.5)
    figure = plt.figure(figsize=(24, 18))

    task_ws = pd.DataFrame(np.load(f'data/results/inter_procedures_opt.npy', allow_pickle=True),
                           index=['RAW', 'NORM', 'MEP', 'TSFRESH', 'TSFRESH_FS'],
                           columns=['RF'])

    ax = sns.heatmap(task_ws, annot=True, fmt=".4f", linewidths=.5, vmin=0, vmax=1, cmap='mako', cbar=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_title('TASK 3 - WS')
    plt.savefig(f'data/plots/task3_ws.png')
    plt.show()

    sns.set(font_scale=2.5)
    figure = plt.figure(figsize=(24, 18))

    task_nl = pd.DataFrame(np.load(f'data/results/inter_procedures_nolat_opt.npy', allow_pickle=True),
                           index=['MEP', 'TSFRESH', 'TSFRESH_FS'],
                           columns=['RF'])

    ax = sns.heatmap(task_nl, annot=True, fmt=".4f", linewidths=.5, vmin=0, vmax=1, cmap='mako', cbar=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_title('TASK 3 - NL')

    plt.savefig(f'data/plots/task3_nl.png')
    plt.show()


def tasks_subplot():
    for proc in ['dcs', 'tcs']:
        task1_sp(proc)
        task2_1_sp(proc)
        task2_2_sp(proc)
    task3()


if __name__ == "__main__":
    pie_chart()
    tasks_subplot()
