import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
import numpy as np
import os

sns.set()


# def smooth(data, sm=2):
#     if sm >= 1:
#         smooth_data = []
#         for d in data:
#             y = np.ones(sm) * 1.0 / sm
#             d = np.convolve(y, d, "same")
#             smooth_data.append(d)
#     return smooth_data


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


mappo_path1 = './data/mappo/result4/record/r_3000.csv'
mappo_path2 = './data/mappo/result5/record/r_3000.csv'
ippo_path1 = './data/ippo/result9/record/r_3000.csv'
ippo_path2 = './data/ippo/result10/record/r_3000.csv'
maddpg_path1 = './data/maddpg/result2/record/r_3000.csv'
maddpg_path2 = './data/maddpg/result3/record/r_3000.csv'

paths = [['./data/mappo/result4/record/r_4500.csv', './data/mappo/result5/record/r_4500.csv',
          './data/mappo/result1/record/r_4500.csv', './data/mappo/result6/record/r_4500.csv'],
         ['./data/ippo/result9/record/r_4500.csv', './data/ippo/result10/record/r_4500.csv'],
         ['./data/maddpg/result2/record/r_ep4500.csv', './data/maddpg/result3/record/r_ep4500.csv',
          './data/maddpg/result1/record/r_ep4500.csv']]

path = os.path.abspath(__file__)


label = ['The proposed', 'IPPO', 'MADDPG']


def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        z = np.ones(len(data))
        y = np.ones(sm) * 1.0
        d = np.convolve(y, data, "same") / np.convolve(y, z, "same")
        smooth_data.append(d)
    return smooth_data


# npy = np.load('./data/ippo/TD3_Ant-v1_0.npy')
# print(npy)

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
    alg_df = pd.concat(alg_df)
    df.append(alg_df)
    # print(pd.concat(alg_df))
df = pd.concat(df)
# df = smooth(df, 9)
print(df)

sns.lineplot(x="Episodes", y="Rewards", hue="algo", style="algo", data=df)
plt.show()
# def get_data():
#     '''获取数据
#     '''
#     basecond = np.array([[18, 20, 19, 18, 13, 4, 1], [20, 17, 12, 9, 3, 0, 0], [20, 20, 20, 12, 5, 3, 0]])
#     cond1 = np.array([[18, 19, 18, 19, 20, 15, 14], [19, 20, 18, 16, 20, 15, 9], [19, 20, 20, 20, 17, 10, 0]])
#     cond2 = np.array([[20, 20, 20, 20, 19, 17, 4], [20, 20, 20, 20, 20, 19, 7], [19, 20, 20, 19, 19, 15, 2]])
#     cond3 = np.array([[20, 20, 20, 20, 19, 17, 12], [18, 20, 19, 18, 13, 4, 1], [20, 19, 18, 17, 13, 2, 0]])
#     return basecond, cond1, cond2, cond3
#
#
# data = get_data()
# print(data)
# label = ['algo1', 'algo2', 'algo3', 'algo4']
# df = []
# for i in range(len(data)):
#     df.append(pd.DataFrame(data[i]).melt(var_name='episode', value_name='loss'))
#     df[i]['algo'] = label[i]
# df = pd.concat(df)  # 合并
# print(df)
# sns.lineplot(x="episode", y="loss", hue="algo", style="algo", data=df)
# plt.title("some loss")
# plt.show()
