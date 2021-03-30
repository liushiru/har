import pandas as pd
import numpy as np
import os
import glob
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from sklearn.cluster import DBSCAN


import config
import scipy.signal as ss
import collections



def get_file_name(dancer_id, move_id, imu_id):
    filename = "dancer_{}_move_{}_{}.csv".format(dancer_id, move_id, imu_id)
    filepath = os.path.join("Data", "RawData", filename)
    return filepath


def get_window_array(df, start):
    window_df = df.loc[start:(start+config.wl-1), :]
    array = window_df.to_numpy().flatten('F')
    return array


def combine_hand_hip():
    for dancer_id in [4, 5]:
        for move_id in range(config.num_moves):
            comb_df = pd.DataFrame()
            for imu_id in ["hand", "hip"]:
                filepath = get_file_name(dancer_id, move_id, imu_id)
                df = pd.read_csv(filepath, header=None, usecols=np.arange(7))
                df = df.iloc[:, :6]
                # if imu_id == "hand":
                #     for i in range(6):
                #         df[i] = 0
                comb_df = pd.concat((comb_df, df), axis=1)
            comb_df.to_csv(get_file_name(dancer_id, move_id, "comb"), index=False, header=False)


def pipe_data():
    combine_hand_hip()
    extract_raw()



def extract_raw():
    raw_files = glob.glob("./Data/RawData/*comb.csv")

    collate_df = pd.DataFrame()

    for file in raw_files:
        raw = []
        info = os.path.basename(file).split('_')
        dancer_id = int(info[1])
        move_id = int(info[3])
        imu_id = info[4]

        file_df = pd.read_csv(file, header=None, usecols=np.arange(config.num_axis))
        total_len = file_df.shape[0]
        num_win = (total_len // (config.wl //2)) - 1

        for win_id in range(num_win):
            win_start = win_id * (config.wl // 2)
            win_arr = get_window_array(file_df, win_start)
            raw.append(win_arr)

        raw_df = pd.DataFrame(raw, columns=np.arange(config.wl * config.num_axis))
        raw_df['dancer'] = dancer_id
        raw_df['label'] = move_id
        collate_df = collate_df.append(raw_df)

    collate_df.to_csv('./Data/RawExtract/raw.csv')
    # labels = pd.read_csv('./Data/RawData/labels.txt', sep=" ", names=['exp_id', 'user_id', 'activity', 'start', 'end'])
    # labels['duration'] = labels['end'] - labels['start']
    # raw_result_df = pd.DataFrame()
    # label_result_df = pd.DataFrame(columns=['label'])
    # raw = []
    # y = []
    # user_ids = []
    # for i in range(len(labels)):
    #     row_act = labels.iloc[i]['activity']
    #     user_id = labels.iloc[i]['user_id']
    #     row = labels.iloc[i, :]
    #
    #     acc_file_name, gyro_file_name = get_file_name(row['exp_id'], row['user_id'])
    #
    #     acc_raw_df = pd.read_csv(acc_file_name, sep=" ", names=[0, 1, 2])
    #     gyro_raw_df = pd.read_csv(gyro_file_name, sep=" ", names=[0, 1, 2])
    #     raw_df = pd.concat((acc_raw_df, gyro_raw_df), axis=1)
    #
    #     total_len = (row['end'] - row['start'])
    #     total_len -= total_len % 64
    #     num_win = (total_len // 64) - 1
    #     act_start = row['start']
    #     for j in range(num_win):
    #         win_start = act_start + j * 64 - 1
    #         win_arr = get_window_array(raw_df, win_start)
    #         raw.append(win_arr)
    #         y.append(row_act)
    #         user_ids.append(user_id)
    # windowed_raw_df = pd.DataFrame(raw, columns=np.arange(128 * 6))
    # windowed_raw_df['label'] = y
    # windowed_raw_df['user_id'] = user_ids
    # windowed_raw_df.to_csv('./Data/RawExtract/raw_user_id.csv')



def down_sample():
    org = pd.read_csv(config.raw_data_path, index_col=0)
    labels = org.loc[: , 'label']
    user_ids = org.loc[:, 'user_id']
    org = org[org.columns[:(0-config.num_label_cols)]]

    roll_mean = org.rolling(2, axis=1).mean()
    roll_mean = roll_mean[roll_mean.columns[1:]]
    down = roll_mean.iloc[:, ::2]
    down.columns = np.arange(64*6)
    down['label'] = labels
    down['user_id'] = user_ids
    down.to_csv('./Data/RawExtract/raw_25hz_id.csv')



def plot_data():
    df = pd.read_csv('Data/RawData/dancer_5_pos_1_hand_2.csv')

    df2 = pd.read_csv('Data/RawData/dancer_5_pos_2_hand.csv')

    plot(df.iloc[:, 0].to_numpy(), label= 'acc_x_hand')
    plot(df.iloc[:, 1].to_numpy(), label='acc_y_hand')
    plot(df.iloc[:, 2].to_numpy(), label='acc_z_hand')
    plot(df.iloc[:, 3].to_numpy(), label='gy_x_hand')
    plot(df.iloc[:, 4].to_numpy(), label='gy_y_hand')
    plot(df.iloc[:, 5].to_numpy(), label='gy_z_hand')

    plt.show()
    plt.legend()
    plot(df2.iloc[:, 0].to_numpy(), label='acc_x_hip')
    plot(df2.iloc[:, 1].to_numpy(), label = 'acc_y_hip')
    plot(df2.iloc[:, 2].to_numpy(), label='acc_z_hip')
    plot(df2.iloc[:, 3].to_numpy())
    plot(df2.iloc[:, 4].to_numpy())
    plot(df2.iloc[:, 5].to_numpy())


def get_velocity(left):
    cs = []
    cum_sum = 0
    i = 0
    stop_time = 4
    while i < (len(left) - stop_time):
        if np.mean(np.abs(left[i: i + stop_time])) < 200:
            print(i)
            cum_sum = 0
            cs.extend([0 for _ in range(stop_time)])
            i += stop_time
        cum_sum += left[i]
        cs.append(cum_sum)
        i+=1
    return cs

from sklearn.cluster import DBSCAN
from scipy.stats import mode
def sync_delay_v1(signal1, signal2):
    # signal12 are numpy array
    # make use of the time difference between the maximum and minimum point in a window
    # return delay in second
    l = []
    for i in range(6):
        l.append(signal1[i].argmax() - signal2[i].argmax())
        l.append(signal1[i].argmin() - signal2[i].argmin())
    c = np.array(l)
    clustering = DBSCAN(eps=3, min_samples=4).fit(c.reshape(-1, 1)).labels_
    if sum(clustering != -1) == 0:
        return np.mean(l)
    c = c[clustering != -1]
    clustering = clustering[clustering != -1]
    mode_cluster = mode(clustering)[0]
    c = c[(clustering == mode_cluster)]
    return c.mean() * 0.05

def is_hand_side_by_side(signal):
    # Jason's implementation
    pass

def sync_delay_v2(signal):
    # return motion start
    start_time = None
    for i in range(29, -1, -1):
        prev = signal[:, i-1]
        curr = signal[:, i]
        if is_hand_side_by_side(prev) and not is_hand_side_by_side(curr):
            start_time = i
            break
    return start_time

def sync_delay_v3(signal):
    # use ml prediction resolution, increase window overlap to 93%
    pass

def sync_delay_v3(signal):
    # Jeremy's implementation which I didn't really get. can test out tmr
    pass




if __name__ == "__main__":
    # pipe_data()
    # extract_raw()
    # combine_hand_hip()

    df1 = pd.read_csv('./Data/RawData/movement_data/dancer_4_pos1_hip.csv', header=None, usecols=np.arange(6))
    df2 = pd.read_csv('./Data/RawData/movement_data/dancer_4_pos2_hip.csv', header=None, usecols=np.arange(6))
    # acc_x = df[0]
    # acc_y = df[1]
    # acc_z = df[2]
    # gy_x = df[2]
    # gy_y = df[2]
    # gy_z = df[2]

    for i in range(0, len(df2) - 30, 15):

        signal1 = df1.iloc[i: i+30, :].to_numpy().T
        signal2 = df2.iloc[i: i+30, :].to_numpy().T
        delay = sync_delay_v1(signal1, signal2)
        print(delay)
        fig, axs = plt.subplots(2)
        for sig in [signal1, signal2]:
            for j in range(6):
                axs[0].plot(signal1[j])
                axs[1].plot(signal2[j])
        plt.show()
        plt.close()

        # plt.clf()
        # l = []
        # m = []
        # d = {}
        # for j in range(6):
        #     diff = df1[j][i:i + 30].max() - df2[j][i:i + 30].min()
        #     d[diff] = [df1[j][i:i + 30].to_numpy(), df2[j][i:i + 30].to_numpy(), j]
        # # od = collections.OrderedDict(sorted(d.items(), reverse=True))
        #
        #
        # # for j in range(6):
        # #     l.append(df1[j][i+10:i + 30+10].to_numpy().argmax() - df2[j][i:i + 30].to_numpy().argmax())
        # #     m.append(df1[j][i+10:i + 30+10].to_numpy().argmin() - df2[j][i:i + 30].to_numpy().argmin())
        # for key in sorted(d, reverse=True)[:]:
        #     l1 = d[key][0]
        #     l2 = d[key][1]
        #     print(d[key][2])
        #     l.append(l1.argmax() - l2.argmax())
        #     m.append(l1.argmin() - l2.argmin())
        #
        # fig, axs = plt.subplots(3)
        # c = np.array(l+m)
        # # plt.figure(i+3)
        # # axs[0].plot(c, np.zeros(len(c)), '.')
        # clustering = DBSCAN(eps=3, min_samples=4).fit(c.reshape(-1,1))
        #
        # cluster = np.unique(clustering.labels_)
        # for clu in np.unique(clustering.labels_):
        #     curr = c[clustering.labels_ == clu]
        #     axs[0].plot(curr, np.zeros(len(curr)), '.', label=clu)
        #     axs[0].legend()
        #     # plt.show()



        # print(l)
        # print(m)
        # print(i,'----------------------------------')

        # plt.figure(i)
    #     for k in range(6):
    #         axs[1].plot(df1[k][i:i+30].to_numpy())
    #     # plt.show()
    #     # plt.figure(i+1)
    #     for k in range(6):
    #         axs[2].plot(df2[k][i:i+30].to_numpy())
    #     plt.show()
    #     plt.close()
    #     pass
    # pass




