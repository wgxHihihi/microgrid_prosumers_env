import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
import numpy as np
import os
import re

sns.set(style='ticks')


# sns.despine(top=True, right=True, left=True, bottom=True)


def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        z = np.ones(len(data))
        y = np.ones(sm) * 1.0
        d = np.convolve(y, data, "same") / np.convolve(y, z, "same")
        smooth_data.append(d)
    return smooth_data[0]


def get_mean_reward(paths, smooth_range):
    df = {}
    for algo, files in paths.items():
        mean_r = []
        for file in files:
            data = pd.read_csv(file).round(2)
            if smooth_range >= 3:
                for key in data.keys():
                    data[key] = smooth(data[key], smooth_range)
            data = data.iloc[:, 1:]
            data_np = data.values[:, :, np.newaxis]
            mean_r.append(data_np)
        mean_rewards = np.stack(mean_r, axis=-1)
        mean_r_value = np.mean(mean_rewards, axis=-1)
        df[algo] = mean_r_value
    return df


def plot_df(df_smooth: dict, df: dict, is_smooth: bool):
    fig = plt.figure(figsize=(8, 5))
    colors = {}
    for algo, data in df_smooth.items():
        total_r = data[:, 0, 0]
        a = plt.plot(range(len(total_r)), list(total_r), label=algo)
        colors[algo] = a[0]._color
    if is_smooth:
        for algo, data in df.items():
            total_r = data[:, 0, 0]
            # print(len(total_r))
            plt.plot(range(len(total_r)), list(total_r), color=colors[algo], linewidth=2, alpha=0.5)
    plt.legend(fontsize=indexsize)
    plt.xticks(fontsize=indexsize)
    plt.xlim(0, 5000)
    plt.ylabel('Rewards', fontsize=fontsize)
    plt.xlabel('Episodes', fontsize=fontsize)
    plt.show()


def plot_load():
    home_a = 'residentialenv/training_data/building_data/hasPV/home1642.csv'
    home_b = 'residentialenv/training_data/building_data/hasPV/home2335.csv'
    home_c = 'residentialenv/training_data/building_data/noPV/home5746.csv'
    home_d = 'residentialenv/training_data/building_data/noPV/home7901.csv'
    home_e = 'residentialenv/training_data/building_data/noPV/home7951.csv'
    dirs = {'home_a': home_a, 'home_b': home_b, 'home_c': home_c, 'home_d': home_d, 'home_e': home_e}
    day_index = 1
    plt.figure(figsize=[8, 5])
    for key, path in dirs.items():
        data = pd.read_csv(path)
        grid_data = data['grid'][day_index * 96:(day_index + 1) * 96]
        plt.plot(range(len(grid_data)), grid_data, label=key)
    plt.legend(fontsize=indexsize)
    plt.xlabel('time (H)', fontsize=fontsize)
    plt.xlim(0, 96)
    plt.xticks(range(0, 97, 8), labels=time_labels, fontsize=indexsize)
    plt.ylabel('Power (kW)', fontsize=fontsize)
    plt.show()


def rewards_plot(paths):
    df = []
    for algo, files in paths.items():
        alg_df = []
        for file in files:
            data = pd.read_csv(file).round(2)
            data = data[['Unnamed: 0', 'total']]
            data['total'] = smooth(data['total'], 19)[0]
            data['algo'] = algo
            data.rename(columns={'Unnamed: 0': 'Episodes', 'total': 'Rewards'}, inplace=True)
            alg_df.append(data)
        alg_df = pd.concat(alg_df, ignore_index=True)
        df.append(alg_df)

    df = pd.concat(df, ignore_index=True)
    print(df)
    sns.lineplot(x="Episodes", y="Rewards", hue="algo", style="algo", data=df)
    plt.xlim(0, ep)
    plt.show()


def get_files_paths(train_logs_dirs, patten):
    paths = {}
    for key, value in train_logs_dirs.items():
        path = []
        for result in os.listdir(value):
            files = os.listdir(value + r'\{}\record'.format(result))
            target = [file for file in files if re.findall(patten, file)]
            if len(target) > 0:
                path.append(value + r'\{}\record\{}'.format(result, target[0]))
        paths[key] = path
    return paths


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dirs = {'mappo': project_dir + r'\train_logs\mappo',
                 'ippo': project_dir + r'\train_logs\ippo',
                 'maddpg': project_dir + r'\train_logs\maddpg',
                 'mappo-sl': project_dir + r'\train_logs\mappo_state_limit',
                 'ippo-sl': project_dir + r'\train_logs\ippo_state_limit',
                 'maddpg-sl': project_dir + r'\train_logs\maddpg_state_limit'
                 }
    fontsize = 14
    indexsize = 12
    ep = 5000
    time_labels = ['0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '0']
    # colors = {'mappo': 'red', 'ippo': 'blue', 'maddpg': 'green'}
    # file_paths = get_files_paths(logs_dirs, 'r_(.*){}\.csv'.format(ep))
    # # rewards_plot(file_paths)
    # print(file_paths)
    # mean_r_smooth = get_mean_reward(file_paths, 31)
    # mean_r = get_mean_reward(file_paths, 0)
    # plot_df(mean_r_smooth, mean_r, True)
    plot_load()
