import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
import numpy as np
import os
import re
import math

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
        mean_r_value[:, 0, 0] = mean_r_value[:, 0, 0] - mean_r_value[:, -1, 0]
        df[algo] = mean_r_value
    return df


def plot_reward_figs():
    file_paths = get_files_paths(logs_dirs, 'r_(.*){}\.csv'.format(ep))
    df_smooth = get_mean_reward(file_paths, 31)
    df_origin = get_mean_reward(file_paths, 0)

    def plot_df(df_s: dict, df: dict, is_smooth: list, axis_label: list, legend: bool):
        plt.figure(figsize=(8, 5))
        colors = {}
        if is_smooth[0]:
            for algo, data in df_s.items():
                total_r = data[:, 0, 0]
                a = plt.plot(range(len(total_r)), list(total_r), label=algo, linewidth=linewidth_1)
                colors[algo] = a[0]._color
        if is_smooth[1]:
            for algo, data in df.items():
                total_r = data[:, 0, 0]
                # print(len(total_r))
                if is_smooth[0]:
                    plt.plot(range(len(total_r)), list(total_r), color=colors[algo], linewidth=linewidth_2, alpha=0.5)
                else:
                    plt.plot(range(len(total_r)), list(total_r), linewidth=linewidth_1, label=algo)
        if legend:
            plt.legend(fontsize=indexsize)
        plt.tick_params(labelsize=indexsize)
        plt.xlim(0, 5000)
        plt.ylabel(axis_label[1], fontsize=fontsize)
        plt.xlabel(axis_label[0], fontsize=fontsize)

    def plot_bar(df: dict, legend: bool, axis_label: list, x_labels: list, width=0.25, space=0.125):
        df_len = len(df)
        start_bias = width * df_len / 2
        x = [0 + j * (width * df_len + space) for j in range(len(list(df.values())[0]))]
        x = np.array(x)
        plt.figure(figsize=[8, 5])
        for i, item in enumerate(df.items()):
            key, vals = item[0], item[1]
            plt.bar(x - start_bias + (2 * i + 1) * width / 2, vals, width=width, label=key)
        plt.xticks(x, x_labels)
        if legend:
            plt.legend(fontsize=indexsize)
        plt.tick_params(labelsize=indexsize)
        if axis_label[0].strip() != '':
            plt.xlabel(axis_label[0], fontsize=fontsize)
        else:
            plt.xticks(fontsize=fontsize)
        if axis_label[1].strip() != '':
            plt.ylabel(axis_label[1], fontsize=fontsize)

    # proposed reward
    proposed_dict_s, proposed_dict = {'mappo': df_smooth['mappo']}, {'mappo': df_origin['mappo']}
    plot_df(proposed_dict_s, proposed_dict, is_smooth=[True, True], axis_label=['Episodes', 'Rewards ($)'],
            legend=False)

    # proposed electricity cost
    proposed_ele_cost_s = {}
    proposed_ele_cost = {}
    for index, home in enumerate('ABCDE'):
        proposed_ele_cost_s['home_' + home] = -proposed_dict_s['mappo'][:, index + 1, np.newaxis]
        proposed_ele_cost['home_' + home] = -proposed_dict['mappo'][:, index + 1, np.newaxis]
    proposed_ele_cost_s['penalty'] = -proposed_dict_s['mappo'][:, 6, np.newaxis]
    proposed_ele_cost['penalty'] = -proposed_dict['mappo'][:, 6, np.newaxis]
    plot_df(proposed_ele_cost_s, proposed_ele_cost, axis_label=['Episodes', 'Cost and penalty ($)'],
            is_smooth=[True, True], legend=True)

    # comparison rewards
    plot_df(df_smooth, df_origin, is_smooth=[True, True], axis_label=['Episodes', 'Rewards ($)'], legend=True)

    # last reward values
    last_reward_df = {}
    for key, vals in df_origin.items():
        last_rewards = df_origin[key][-1]
        ele_cost = -last_rewards[1:-2, 0].sum()
        p_lim = -last_rewards[-2, 0]
        last_reward_df[key] = [ele_cost, p_lim]
    plot_bar(last_reward_df, axis_label=['', 'Cost and penalty ($)'], legend=True,
             x_labels=['ele_cost', 'p_lim'],
             width=0.125, space=0.25)
    plt.show()


def plot_load_figs():
    home_a = 'residentialenv/training_data/building_data/noPV/home5746.csv'
    home_b = 'residentialenv/training_data/building_data/noPV/home7901.csv'
    home_c = 'residentialenv/training_data/building_data/noPV/home7951.csv'
    home_d = 'residentialenv/training_data/building_data/hasPV/home1642.csv'
    home_e = 'residentialenv/training_data/building_data/hasPV/home2335.csv'
    dirs = {'home_A': home_a, 'home_B': home_b, 'home_C': home_c, 'home_D': home_d, 'home_E': home_e}
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


def plot_record_figs():
    best_res_paths = get_beat_result(logs_dirs)

    def plot_after_scheduling(path):
        data = pd.read_csv(path)
        p_net = data['p_net']
        tou = data['rtp']
        fig = plt.figure(figsize=[8, 5])
        ax1 = fig.add_subplot()
        p_net_line = ax1.plot(range(len(p_net)), p_net, color='royalblue', linewidth=linewidth_1, label='P_net')
        p_lim_line = ax1.hlines(y=14, xmin=0, xmax=96, color='deeppink', label='power_limitation', linestyle='--',
                                linewidth=linewidth_1)
        ax1.tick_params(labelsize=indexsize)
        ax1.set_xlim(0, 96)
        ax1.set_ylim(0, 24)
        ax1.set_yticks(range(0, 25, 2))
        ax1.set_xlabel('time (H)', fontsize=fontsize)
        ax1.set_ylabel('power (kW)', fontsize=fontsize)

        ax2 = ax1.twinx()
        tou_line = ax2.step(range(len(tou)), tou, color='g', linewidth=linewidth_1, label='TOU')
        ax2.set_ylabel('time-of-use price ($)', fontsize=fontsize)
        ax2.set_ylim(0, 0.6)
        ax2.tick_params(labelsize=indexsize)

        lines = [p_net_line[0], tou_line[0], p_lim_line]
        print(lines)

        plt.legend(lines, [l.get_label() for l in lines], fontsize=indexsize, loc=2)
        plt.xticks(time_index, time_labels)

    def plot_soc(path):
        data = pd.read_csv(path)
        plt.figure(figsize=[8, 5])
        for index, label in enumerate('ABCDE'):
            soc = data['soc' + str(index + 1)]
            plt.plot(range(len(soc)), soc * 100, label='home' + label, linewidth=linewidth_1)
        plt.xticks(time_index, time_labels)
        plt.xlabel('Time (H)', fontsize=fontsize)
        plt.ylabel('State-of-charge (%)', fontsize=fontsize)
        plt.legend(fontsize=indexsize)

    def plt_home_power(after_path: str, before_path: str):
        data_after_schedule = pd.read_csv(after_path)
        data_before_schedule = pd.read_csv(before_path)
        rtp = data_after_schedule['rtp']

        y_lim = [[-2, 6], [-2, 10], [-2, 8], [-5, 5], [-5, 10], [0, 24]]
        before, after = {}, {}
        for index, home in enumerate('ABCDE'):
            power_before = data_before_schedule['power_' + str(index + 1)]
            power_after = data_after_schedule['power_' + str(index + 1)]
            before['home_' + home] = power_before
            after['home_' + home] = power_after
        # before['p_mu'] = data_before_schedule['p_net']
        # after['p_mu'] = data_after_schedule['p_net']
        plt.figure(figsize=[8, 8])
        for index, key_val in enumerate(before.items()):
            key = key_val[0]
            power_before = key_val[1]
            power_after = after[key]
            ax1 = plt.subplot(5, 1, index + 1)

            p_after = ax1.plot(range(len(power_after)), power_after, label='P_rb (after scheduling)',
                               linewidth=linewidth_1)
            p_before = ax1.plot(range(len(power_before)), power_before, label='P_rb (before scheduling)',
                                linewidth=linewidth_1)
            ax1.set_ylim(y_lim[index])
            ax1.set_ylabel('power (kW)', fontsize=fontsize)
            ax1.tick_params(labelsize=indexsize)
            ax2 = ax1.twinx()
            rtp_line = ax2.step(range(len(rtp)), rtp, color='seagreen', label='rtp', linewidth=linewidth_1,
                                linestyle='--', alpha=0.9)
            ax2.set_ylim(0, 0.6)
            ax2.set_ylabel('rtp ($)', fontsize=fontsize)
            # ax2.set_xlabel('rtp ($)', fontsize=fontsize)
            ax2.tick_params(labelsize=indexsize)

            lines = [p_after[0], p_before[0], rtp_line[0]]
            # plt.legend(lines, [l.get_label() for l in lines])
            plt.title(key, fontsize=indexsize)
            plt.xlim(0, 96)
            plt.xticks(time_index, time_labels)

            ax = plt.gca()

            ax.axes.xaxis.set_ticklabels([])
            # ax.axes.yaxis.set_ticklabels([])
        ra = range(0, 97, 16)
        la = ['0', '6', '12', '18', '0']
        plt.xticks(time_index, time_labels)
        # plt.legend(lines, [l.get_label() for l in lines])
        # plt.gca()
        plt.xlabel('time (H)', fontsize=fontsize)
        plt.tight_layout()
        # plt.subplots_adjust(wspace=0.5, hspace=0.1)

    # plot_after_scheduling(best_res_paths['mappo'])
    # plot_soc(best_res_paths['mappo'])

    without_dr_path = './train_logs/witout_dr/record_without_dr_1.csv'
    plt_home_power(best_res_paths['mappo'], without_dr_path)
    # plot_after_scheduling(without_dr_path)
    plt.show()


def get_beat_result(train_logs_dirs):
    record_paths = {}
    r_path = get_files_paths(train_logs_dirs, 'r_(.*){}\.csv'.format(ep))

    for key, paths in r_path.items():
        last_rewards = {}
        for path in paths:
            last_reward = pd.read_csv(path).iloc[-1, 1]
            last_rewards[path] = last_reward
        # find out the best result
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

    # plot figures
    plot_reward_figs()
    # plot_load_figs()
    # plot_record_figs()
