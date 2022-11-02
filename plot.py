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


def plot_reward_figs():
    file_paths = get_files_paths(logs_dirs, 'r_(.*){}\.csv'.format(ep))
    df_smooth = get_mean_reward(file_paths, 31)
    df_origin = get_mean_reward(file_paths, 0)

    def plot_df(df_s: dict, df: dict, is_smooth: bool, fig_num: int, legend: bool):
        plt.figure(fig_num, figsize=(8, 5))
        colors = {}
        for algo, data in df_s.items():
            total_r = data[:, 0, 0]
            a = plt.plot(range(len(total_r)), list(total_r), label=algo, linewidth=linewidth_1)
            colors[algo] = a[0]._color
        if is_smooth:
            for algo, data in df.items():
                total_r = data[:, 0, 0]
                # print(len(total_r))
                plt.plot(range(len(total_r)), list(total_r), color=colors[algo], linewidth=linewidth_2, alpha=0.5)
        if legend:
            plt.legend(fontsize=indexsize)
        plt.tick_params(labelsize=indexsize)
        plt.xlim(0, 5000)
        plt.ylabel('Rewards', fontsize=fontsize)
        plt.xlabel('Episodes', fontsize=fontsize)

    # proposed reward
    proposed_dict_s, proposed_dict = {'mappo': df_smooth['mappo']}, {'mappo': df_origin['mappo']}
    plot_df(proposed_dict_s, proposed_dict, is_smooth=True, fig_num=1, legend=False)
    # comparison rewards
    plot_df(df_smooth, df_origin, is_smooth=True, fig_num=2, legend=True)
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
        plt.plot(range(len(grid_data)), grid_data, label=key, linewidth=linewidth_1)
    plt.legend(fontsize=indexsize)
    plt.xlabel('time (H)', fontsize=fontsize)
    plt.xlim(0, 96)
    plt.xticks(range(0, 97, 8), labels=time_labels, fontsize=indexsize)
    plt.tick_params(labelsize=indexsize)
    plt.ylabel('Power (kW)', fontsize=fontsize)
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


def plot_record():
    best_res_paths = get_beat_result(logs_dirs)

    def plot_after_scheduling(path):
        data = pd.read_csv(path)
        p_net = data['p_net']
        tou = data['rtp']
        fig = plt.figure(figsize=[8, 5])
        ax1 = fig.add_subplot()
        p_net_line = ax1.plot(range(len(p_net)), p_net, color='royalblue', linewidth=linewidth_1, label='P_net')

        ax1.tick_params(labelsize=indexsize)
        ax1.set_xlim(0, 96)
        ax1.set_ylim(0, 25)
        ax1.set_xlabel('time (H)', fontsize=fontsize)
        ax1.set_ylabel('power (kW)', fontsize=fontsize)

        ax2 = ax1.twinx()
        tou_line = ax2.step(range(len(tou)), tou, color='g', linewidth=linewidth_1, label='TOU')
        ax2.set_ylabel('time-of-use price ($)', fontsize=fontsize)
        ax2.set_ylim(0, 0.6)
        lines = [p_net_line[0], tou_line[0]]
        print(lines)
        plt.legend(lines, [l.get_label() for l in lines], fontsize=indexsize,loc=2)
        plt.xticks(time_index, time_labels)

    plot_after_scheduling(best_res_paths['mappo'])
    without_dr_path = './train_logs/witout_dr/record_without_dr_1.csv'
    plot_after_scheduling(without_dr_path)
    plt.show()


def get_beat_result(train_logs_dirs):
    record_paths = {}
    r_path = get_files_paths(train_logs_dirs, 'r_(.*){}\.csv'.format(ep))

    for key, paths in r_path.items():
        last_rewards = {}
        for path in paths:
            last_reward = pd.read_csv(path).iloc[-1, 1]
            last_rewards[path] = last_reward

        best_res_path = max(last_rewards, key=lambda x: last_rewards[x])
        # find out the best result folder
        file_folder = os.path.dirname(best_res_path)
        files = os.listdir(file_folder)
        # find out the best result record file
        target = [file for file in files if re.findall('record_{}(.*)\.csv'.format(ep), file)]
        record_paths[key] = file_folder + r'\{}'.format(target[0])
    return record_paths


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dirs = {'mappo': project_dir + r'\train_logs\mappo',
                 'ippo': project_dir + r'\train_logs\ippo',
                 'maddpg': project_dir + r'\train_logs\maddpg',
                 'mappo-sl': project_dir + r'\train_logs\mappo_state_limit',
                 'ippo-sl': project_dir + r'\train_logs\ippo_state_limit',
                 'maddpg-sl': project_dir + r'\train_logs\maddpg_state_limit'
                 }
    fontsize = 16
    indexsize = 14
    linewidth_1 = 2
    linewidth_2 = 2
    ep = 5000
    time_labels = ['0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '0']
    time_index = range(0, 97, 8)
    colors = {'mappo': 'red', 'ippo': 'blue', 'maddpg': 'green'}
    # plot_reward_figs()
    # plot_load()
    plot_record()
