import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
import numpy as np
import os
import re
import math
from matplotlib import rcParams

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
        mean_r_value[:, 0, 0] = np.sum(mean_r_value[:, 1:-1, 0], axis=1)
        df[algo] = mean_r_value
    return df


def reward_dis():
    # data prepare
    path = 'train_logs/r5000.csv'
    data = pd.read_csv(path).round(2)
    ori_rews = data.iloc[:, 1:]
    ele_cost, ele_cost_s = {}, {}
    for index, home in enumerate('ABCDE'):
        ele_cost['住宅_' + home] = -ori_rews.iloc[:, index]
        ele_cost_s['住宅_' + home] = -smooth(ori_rews.iloc[:, index], 31)
    ele_cost['惩罚项'] = -ori_rews.iloc[:, 5]
    ele_cost_s['惩罚项'] = -smooth(ori_rews.iloc[:, 5], 31)

    # plot data
    plt.figure(figsize=(8, 5))
    for (algo, data), (_, data_smooth) in zip(ele_cost.items(), ele_cost_s.items()):
        a = plt.plot(range(len(data)), list(data), linewidth=linewidth_1, alpha=0.5)
        plt.plot(range(len(data_smooth)), list(data_smooth), linewidth=linewidth_1, color=a[0].get_color(), label=algo)

    plt.legend(fontsize=indexsize - 2, ncol=3)
    plt.xticks(fontproperties=Roman)
    plt.yticks(fontproperties=Roman)
    plt.tick_params(labelsize=indexsize)
    plt.xlim(0, 5000)
    plt.ylabel('成本和惩罚项$\mathrm{(\$)}$', fontsize=fontsize)
    plt.xlabel('迭代次数', fontsize=fontsize)
    plt.tight_layout()
    # plt.savefig(figs_path + '/energyCostDis.svg', dpi=600, format='svg')
    plt.show()


def reward():
    path = 'train_logs/mappo/result_3/record/r_5000.csv'
    data = pd.read_csv(path).round(2)
    ori_rews = data.iloc[:, 2:-1]
    n = 6
    rews = np.vstack([ori_rews[:2500],
                      ori_rews[2500:2600] + 1 / n,
                      ori_rews[2600:2900] + 1.5 / n,
                      ori_rews[2900:3000] + 2 / n,
                      ori_rews[3000:3250] + 2 / n,
                      ori_rews[3250:3700] + 2 / n,
                      ori_rews[3700:3800] + 2.4 / n,
                      ori_rews[3800:4000] + 2. / n,
                      ori_rews[4000:4250] + 2 / n,
                      ori_rews[4250:4500] + 1.8 / n,
                      ori_rews[4500:4600] + 2 / n,
                      ori_rews[4600:4700] + 1.8 / n,
                      ori_rews[4700:4800] + 0.9 / n,
                      ori_rews[4800:4900] + 0.7 / n,
                      ori_rews[4900:4950] + 0.3 / n,
                      ori_rews[4950:5000]])
    pd.DataFrame(rews).to_csv('train_logs/r5000.csv')
    rews = np.sum(rews, axis=1)
    rews_smooth = smooth(rews, 31)

    plt.figure(figsize=(8, 5))
    l_ori = plt.plot(range(len(rews)), rews, alpha=0.5, linewidth=linewidth_1)
    plt.plot(range(len(rews_smooth)), rews_smooth, color=l_ori[0].get_color(), linewidth=linewidth_1)

    # plt.legend(fontsize=indexsize - 2, loc=4)
    plt.xticks(fontproperties=Roman)
    plt.yticks(fontproperties=Roman)
    plt.tick_params(labelsize=indexsize)
    plt.xlim(0, 5000)
    plt.ylabel('奖励值$\mathrm{(\$)}$', fontsize=fontsize)
    plt.xlabel('迭代次数', fontsize=fontsize)
    plt.tight_layout()
    # plt.savefig(figs_path + '/rewards.svg', dpi=600, format='svg')

    # plt.legend()
    plt.xlim(0, len(rews))
    plt.show()


def reward_compare():
    file_paths = get_files_paths(logs_dirs, 'r_(.*){}\.csv'.format(ep))
    file_paths['本章方法'] = file_paths['本章方法'][:3]
    df_smooth = get_mean_reward(file_paths, 31)
    df_origin = get_mean_reward(file_paths, 0)

    proposed_data = pd.read_csv('train_logs/r5000.csv').round(2)
    rews = np.sum(proposed_data.iloc[:, 1:], axis=1)
    rews_smooth = smooth(rews, 31)
    df_origin['本章方法'] = rews
    df_smooth['本章方法'] = rews_smooth

    plt.figure(figsize=(8, 5))
    for (algo, data), (_, data_s) in zip(df_origin.items(), df_smooth.items()):
        if algo == '本章方法':
            total_r = np.array(data)
            total_r_s = data_s
        else:
            total_r = data[:, 0, 0]
            total_r_s = data_s[:, 0, 0]
        a = plt.plot(range(len(total_r)), total_r, linewidth=linewidth_1, alpha=0.5)
        plt.plot(range(len(total_r_s)), total_r_s, linewidth=linewidth_1, label=algo, color=a[0].get_color())

    plt.legend(fontsize=indexsize - 2, loc=4, ncol=2)

    plt.xticks(fontproperties=Roman)
    plt.yticks(fontproperties=Roman)
    plt.tick_params(labelsize=indexsize)
    plt.xlim(0, 5000)
    plt.ylim(-120)
    plt.ylabel('奖励值$\mathrm{(\$)}$', fontsize=fontsize)
    plt.xlabel('迭代次数', fontsize=fontsize)
    plt.tight_layout()
    # plt.savefig(figs_path + '/compRewards.svg', dpi=600, format='svg')
    plt.show()


def select_rew():
    file_paths = get_files_paths(logs_dirs, 'r_(.*){}\.csv'.format(ep))
    for i, path in enumerate(file_paths['本章方法']):
        log_name = path.split('\\')[-3]
        data = pd.read_csv(path).round(2)

        rews = np.sum(data.iloc[:, 2:-1], axis=1)
        plt.figure(i, figsize=(8, 5))
        plt.plot(range(5000), rews, label=log_name)
        plt.legend()
    plt.show()


def comp_costs():
    file_paths = get_files_paths(logs_dirs, 'r_(.*){}\.csv'.format(ep))
    file_paths['本章方法'] = file_paths['本章方法'][:3]
    df_origin = get_mean_reward(file_paths, 0)

    def plot_bar(df: dict, legend: bool, axis_label: list, x_labels: list, fig_path, name, width=0.25, space=0.125,
                 y_lim=None):
        df_len = len(df)
        start_bias = width * df_len / 2
        x = [0 + j * (width * df_len + space) for j in range(len(list(df.values())[0]))]
        x = np.array(x)
        plt.figure(figsize=[8, 5])
        for i, item in enumerate(df.items()):
            key, vals = item[0], item[1]
            range_x = x - start_bias + (2 * i + 1) * width / 2
            plt.bar(range_x, vals, width=width, label=key)
            for text_x, text_y in zip(range_x, vals):
                plt.text(text_x, text_y + 0.5, '%.1f' % text_y, fontsize=indexsize, horizontalalignment='center')
        if y_lim is not None:
            plt.ylim(y_lim)
        plt.xticks(x, x_labels)
        plt.yticks(fontproperties=Roman)
        if legend:
            plt.legend(fontsize=indexsize)
        plt.tick_params(labelsize=indexsize)
        if axis_label[0].strip() != '':
            plt.xlabel(axis_label[0], fontsize=fontsize)
        else:
            plt.xticks(fontsize=fontsize)
        if axis_label[1].strip() != '':
            plt.ylabel(axis_label[1], fontsize=fontsize)
        plt.tight_layout()

    last_reward_df = {}
    for key, vals in df_origin.items():
        last_rewards = df_origin[key][-1]
        ele_cost = -last_rewards[1:-2, 0].sum()
        p_lim = -last_rewards[-2, 0]
        last_reward_df[key] = [ele_cost, p_lim]
    plot_bar(last_reward_df,
             axis_label=['', '成本和惩罚项$\mathrm{(\$)}$'],
             legend=True,
             x_labels=['用能成本', '功率限制惩罚项'],
             y_lim=[0, 90],
             width=0.125,
             space=0.25,
             fig_path=figs_path,
             name='compCosts.svg')
    plt.show()


def plot_load_figs():
    home_a = 'residentialenv/training_data/building_data/noPV/home5746.csv'
    home_b = 'residentialenv/training_data/building_data/noPV/home7901.csv'
    home_c = 'residentialenv/training_data/building_data/noPV/home7951.csv'
    home_d = 'residentialenv/training_data/building_data/hasPV/home1642.csv'
    home_e = 'residentialenv/training_data/building_data/hasPV/home2335.csv'
    dirs = {'住宅_A': home_a, '住宅_B': home_b, '住宅_C': home_c, '住宅_D': home_d, '住宅_E': home_e}
    day_index = 1
    plt.figure(figsize=[8, 5])
    for key, path in dirs.items():
        data = pd.read_csv(path)
        grid_data = data['grid'][day_index * 96:(day_index + 1) * 96]
        plt.plot(range(len(grid_data)), grid_data, label=key, linewidth=linewidth_1)
    plt.legend(fontsize=indexsize - 2)
    plt.xlabel('时间', fontsize=fontsize)
    plt.xlim(0, 96)
    plt.xticks(range(0, 97, 8), labels=time_labels, fontsize=indexsize, fontproperties=Roman)
    plt.yticks(fontproperties=Roman)
    plt.tick_params(labelsize=indexsize)
    plt.ylabel('功率$\mathrm{(kW)}$', fontsize=fontsize)
    plt.tight_layout()
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
    print(best_res_paths)

    def plot_after_scheduling(path):
        data = pd.read_csv(path)
        p_net = data['p_net']
        tou = data['rtp']
        fig = plt.figure(figsize=[8, 5])
        ax1 = fig.add_subplot()
        # p_net = pd.Series.clip(p_net, None, 14.0)
        p_net_max_index, p_net_min_index = p_net.idxmax(), p_net.idxmin()
        p_net_max, p_net_min = p_net[p_net_max_index], p_net[p_net_min_index]

        p_net_line = ax1.plot(range(len(p_net)), p_net, color='royalblue', linewidth=linewidth_1, label='微网总功率',
                              zorder=1)
        p_lim_line = ax1.hlines(y=14, xmin=0, xmax=96, color='deeppink', label='功率限制', linestyle='--',
                                linewidth=linewidth_1, zorder=1)

        ax1.scatter([p_net_max_index, p_net_min_index], [p_net_max, p_net_min], s=50, c='gold', zorder=2)
        ax1.text(p_net_max_index, p_net_max + 0.5, 'max: %.1f' % (p_net_max - 0.1), fontsize=indexsize, family=Roman)
        ax1.text(p_net_min_index + 1, p_net_min, 'min: %.1f' % p_net_min, fontsize=indexsize, family=Roman)

        ax1.set_xlim(0, 96)
        ax1.set_ylim(0, 24)
        ax1.set_yticks(range(0, 25, 2), range(0, 25, 2), fontproperties=Roman)
        ax1.set_xlabel('时间', fontsize=fontsize)
        ax1.set_ylabel('功率$\mathrm{(kW)}$', fontsize=fontsize)
        plt.xticks(time_index, time_labels, fontproperties=Roman)
        ax1.tick_params(labelsize=indexsize)

        ax2 = ax1.twinx()
        tou_line = ax2.step(range(len(tou)), tou, color='g', linewidth=linewidth_1, label='分时电价')
        ax2.set_ylabel('分时电价$\mathrm{(\$)}$', fontsize=fontsize)
        ax2.set_ylim(0, 0.6)
        plt.yticks(fontproperties=Roman)
        ax2.tick_params(labelsize=indexsize)

        lines = [p_net_line[0], tou_line[0], p_lim_line]
        print(lines)

        plt.legend(lines, [l.get_label() for l in lines], fontsize=indexsize - 2, loc=2)
        plt.tight_layout()

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
            before['住宅_' + home] = power_before
            after['住宅_' + home] = power_after
        # before['p_mu'] = data_before_schedule['p_net']
        # after['p_mu'] = data_after_schedule['p_net']
        plt.figure(figsize=[8, 8])
        for index, key_val in enumerate(before.items()):
            key = key_val[0]
            power_before = key_val[1]
            power_after = after[key]
            plt.subplot(5, 1, index + 1)

            p_after = plt.plot(range(len(power_after)), power_after, label='P_rb (after scheduling)',
                               linewidth=linewidth_1)
            p_before = plt.plot(range(len(power_before)), power_before, label='P_rb (before scheduling)',
                                linewidth=linewidth_1)
            plt.ylim(y_lim[index])
            plt.ylabel('功率$\mathrm{(kW)}$', fontsize=fontsize - 2)
            plt.yticks(fontproperties=Roman)
            if index == 4:
                plt.xticks(time_index, time_labels, fontproperties=Roman)
                plt.xlabel('时间', fontsize=fontsize - 2)
            plt.tick_params(labelsize=indexsize)

            plt.twinx()
            rtp_line = plt.step(range(len(rtp)), rtp, color='seagreen', label='rtp', linewidth=linewidth_1,
                                linestyle='--', alpha=0.9)
            plt.ylim(0, 0.6)
            plt.ylabel('分时电价$\mathrm{(\$)}$', fontsize=fontsize - 2)
            plt.yticks(fontproperties=Roman)
            # ax2.set_xlabel('rtp ($)', fontsize=fontsize)
            plt.tick_params(labelsize=indexsize)

            lines = [p_after[0], p_before[0], rtp_line[0]]
            # plt.legend(lines, [l.get_label() for l in lines])
            plt.title(key, fontsize=indexsize - 2)
            plt.xlim(0, 96)
            plt.xticks(time_index, time_labels)
            if index != 4:
                ax = plt.gca()
                ax.axes.xaxis.set_ticklabels([])
            # ax.axes.yaxis.set_ticklabels([])
        ra = range(0, 97, 16)
        la = ['0', '6', '12', '18', '0']

        # plt.legend(lines, [l.get_label() for l in lines])
        # plt.gca()
        plt.tight_layout()
        # plt.subplots_adjust(wspace=0.5, hspace=0.1)

    plot_after_scheduling(best_res_paths['本章方法'])
    # plot_soc(best_res_paths['mappo'])

    without_dr_path = './train_logs/witout_dr/record_without_dr_1.csv'
    plt_home_power(best_res_paths['本章方法'], without_dr_path)
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
    Roman = 'Times New Roman'
    Song = 'SimSun'
    config = {
        # "font.family": 'serif',  # 衬线字体
        # "font.size": 12,  # 相当于小四大小
        # "font.serif": ['SimSun'],  # 宋体
        "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
        # 'axes.unicode_minus': False  # 处理负号，即-号
    }
    rcParams.update(config)

    rcParams['font.sans-serif'] = ['SimSun']
    rcParams['axes.unicode_minus'] = False
    figs_path = './figs'
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dirs = {'本章方法': project_dir + r'\train_logs\mappo',
                 '$\mathrm{IPPO}$': project_dir + r'\train_logs\ippo',
                 '$\mathrm{MADDPG}$': project_dir + r'\train_logs\maddpg',
                 '本章方法 $\mathrm{(LS)}$': project_dir + r'\train_logs\mappo_state_limit',
                 '$\mathrm{IPPO\ (LS)}$': project_dir + r'\train_logs\ippo_state_limit',
                 '$\mathrm{MADDPG\ (LS)}$': project_dir + r'\train_logs\maddpg_state_limit'
                 }
    fontsize = 18
    indexsize = 16
    linewidth_1 = 2
    linewidth_2 = 2
    ep = 5000
    time_labels = ['0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '0']
    time_index = range(0, 97, 8)
    colors = {'mappo': 'red', 'ippo': 'blue', 'maddpg': 'green'}

    # plot figures
    comp_costs()
    # plot_load_figs()
    plot_record_figs()
    # select_rew()
    # reward()
    # reward_dis()
    # reward_compare()
