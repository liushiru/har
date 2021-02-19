import pandas as pd
import numpy as np


def get_file_name(exp_id, user_id):
    base = '_exp' + str(exp_id).zfill(2) + '_user' + str(user_id).zfill(2) + '.txt'
    front = './Data/RawData/'
    return front + 'acc' + base, front + 'gyro' + base


def get_window_array(df, start):
    window_df = df.loc[start:start+127, :]
    array = window_df.to_numpy().flatten('F')
    return array


def extract_raw():
    labels = pd.read_csv('./Data/RawData/labels.txt', sep=" ", names=['exp_id', 'user_id', 'activity', 'start', 'end'])
    labels['duration'] = labels['end'] - labels['start']
    raw_result_df = pd.DataFrame()
    label_result_df = pd.DataFrame(columns=['label'])
    raw = []
    y = []
    for i in range(len(labels)):
        row_act = labels.iloc[i]['activity']
        row = labels.iloc[i, :]

        acc_file_name, gyro_file_name = get_file_name(row['exp_id'], row['user_id'])

        acc_raw_df = pd.read_csv(acc_file_name, sep=" ", names=[0, 1, 2])
        gyro_raw_df = pd.read_csv(gyro_file_name, sep=" ", names=[0, 1, 2])
        raw_df = pd.concat((acc_raw_df, gyro_raw_df), axis=1)

        total_len = (row['end'] - row['start'])
        total_len -= total_len % 64
        num_win = (total_len // 64) - 1
        act_start = row['start']
        for j in range(num_win):
            win_start = act_start + j * 64 - 1
            win_arr = get_window_array(raw_df, win_start)
            raw.append(win_arr)
            y.append(row_act)
    windowed_raw_df = pd.DataFrame(raw, columns=np.arange(128 * 6))
    windowed_raw_df['label'] = y
    windowed_raw_df.to_csv('./Data/RawExtract/windowed_raw.csv')



def down_sample():
    org = pd.read_csv('./Data/RawExtract/windowed_raw.csv', index_col=0)
    labels = org.loc[:,'label']
    org = org[org.columns[:-1]]

    roll_mean = org.rolling(2, axis=1).mean()
    roll_mean = roll_mean[roll_mean.columns[1:]]
    down = roll_mean.iloc[:, ::2]
    down.columns = np.arange(64*6)
    down['label'] = labels
    down.to_csv('./Data/RawExtract/raw_25hz.csv')


if __name__ == "__main__":
    # extract_raw()
    down_sample()



