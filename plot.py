import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
import numpy as np
import os

sns.set()


def reward(path):
    reward_data = pd.read_csv(path)
    sns.lineplot(x='index', y='total', data=reward_data, alpha=0.5)
    # sns.lineplot(x='index', y='agent1', data=reward_data, alpha=0.5)
    tmp_smooth1 = savgol_filter(reward_data['total'], 41, 3)
    # tmp_smooth2 = savgol_filter(reward_data['agent1'], 41, 3)
    plt.plot(reward_data['index'], tmp_smooth1)
    # plt.plot(reward_data['index'], tmp_smooth2)
    plt.xlim(0, 5000)
    # plt.ylim(-100, 10)
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.show()


def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        z = np.ones(len(data))
        y = np.ones(sm) * 1.0
        d = np.convolve(y, data, "same") / np.convolve(y, z, "same")
        smooth_data.append(d)
    return smooth_data


def rewards_plot():
    df = []
    for i in range(len(paths)):
        alg_df = []
        for j in range(len(paths[i])):
            data = pd.read_csv(paths[i][j]).round(2)
            data = data[['Unnamed: 0', 'total']]
            # print (smooth(data['total'],9))
            data['total'] = smooth(data['total'], 19)[0]
            data['algo'] = label[i]
            data.rename(columns={'Unnamed: 0': 'Episodes', 'total': 'Rewards'}, inplace=True)
            alg_df.append(data)
        alg_df = pd.concat(alg_df, ignore_index=True)
        df.append(alg_df)

    df = pd.concat(df, ignore_index=True)

    sns.lineplot(x="Episodes", y="Rewards", hue="algo", style="algo", data=df)
    plt.show()


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    mappo_path1 = project_dir + '/train_log/result_mappo12/record/r_5000.csv'
    mappo_path2 = project_dir + '/train_log/result_ippo22/record/r_5000.csv'
    ippo_path1 = './data/ippo/result9/record/r_3000.csv'
    ippo_path2 = './data/ippo/result10/record/r_3000.csv'
    maddpg_path1 = './data/maddpg/result2/record/r_3000.csv'
    maddpg_path2 = './data/maddpg/result3/record/r_3000.csv'

    paths = [[mappo_path1, mappo_path2]]

    path = os.path.abspath(__file__)

    label = ['The proposed', 'IPPO', 'MADDPG']

    rewards_plot()
