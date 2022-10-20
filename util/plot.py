import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import clf_names


def heatmap():
    # for x in ['dcs', 'tcs']:
    #     task1 = np.load(f'data/results/intra_patient_{x}.npy', allow_pickle=True)
    #     rep, sub, clf = task1.shape
    #     hm = np.zeros((rep, clf))
    #     for i in range(rep):
    #         for j in range(clf):
    #             hm[i, j] = task1[i, :, j].mean()
    #     task1 = pd.DataFrame(hm,
    #                          index=['RAW', 'NORM', 'MEP', 'TSFRESH', 'TSFRESH_FS'],
    #                          columns=clf_names)
    #     fig = plt.figure(figsize=(24, 18))
    #     sns.set(font_scale=2.5)
    #     sns.despine(bottom=True, left=True)
    #     ax = sns.heatmap(task1, annot=True, fmt=".4f", linewidths=.5, vmin=0, vmax=1, cmap='mako')
    #     ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    #     plt.title(f'Task 1 - Intra Patient - {x.upper()}')
    #     plt.savefig(f'data/plots/task1_{x}.png')
    #     plt.show()

    for x in ['dcs', 'tcs']:
        task2_1 = pd.DataFrame(np.load(f'data/results/inter_patient2_{x}.npy', allow_pickle=True),
                             index=['RAW', 'NORM', 'MEP', 'TSFRESH', 'TSFRESH_FS'],
                             columns=clf_names)
        fig = plt.figure(figsize=(24, 18))
        sns.set(font_scale=2.5)
        sns.despine(bottom=True, left=True)
        ax = sns.heatmap(task2_1, annot=True, fmt=".4f", linewidths=.5, vmin=0, vmax=1, cmap='mako')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        plt.title(f'Task 2 - Inter Patient - {x.upper()}')
        plt.savefig(f'data/plots/task2_2_{x}.png')
        plt.show()

        task2_2 = pd.DataFrame(np.load(f'data/results/inter_patient4_{x}.npy', allow_pickle=True),
                               index=['RAW', 'NORM', 'MEP', 'TSFRESH', 'TSFRESH_FS'],
                               columns=clf_names)
        fig = plt.figure(figsize=(24, 18))
        sns.set(font_scale=2.5)
        sns.despine(bottom=True, left=True)
        ax = sns.heatmap(task2_2, annot=True, fmt=".4f", linewidths=.5, vmin=0, vmax=1, cmap='mako')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        plt.title(f'Task 2 - Inter Patient - {x.upper()}')
        plt.savefig(f'data/plots/task2_4_{x}.png')
        plt.show()

    # task3 = pd.DataFrame(np.load(f'data/results/inter_procedures.npy', allow_pickle=True),
    #                        index=['RAW', 'NORM', 'MEP', 'TSFRESH', 'TSFRESH_FS'],
    #                        columns=clf_names)
    # fig = plt.figure(figsize=(24, 18))
    # sns.set(font_scale=2.5)
    # sns.despine(bottom=True, left=True)
    # ax = sns.heatmap(task3, annot=True, fmt=".4f", linewidths=.5, vmin=0, vmax=1, cmap='mako')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    # plt.title(f'Task 3 - Inter Procedures')
    # plt.savefig(f'data/plots/task3.png')
    # plt.show()

heatmap()