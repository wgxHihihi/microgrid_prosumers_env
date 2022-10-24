import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import savgol_filter

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


path2 = '../../train_logs/result3/record/r_3000.csv'
path1 = '../../train_logs/result4/record/r_3000.csv'
reward1 = pd.read_csv(path1)
reward2 = pd.read_csv(path2)
df = reward1.append(reward2)
# df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
df.index = range(len(df))
print(df)
plt.figure(1, [5, 8])
sns.lineplot(x='Unnamed: 0', y='total', data=df)
plt.savefig('./total.pdf')
# reward(path)
