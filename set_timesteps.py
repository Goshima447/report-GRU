#!/usr/bin/env python
# coding: utf-8


import numpy as np

# 保持する過去データ数を"timesteps"にて設定（1step = 1時間分）
# 使用例： 
# import set_timesteps
# x_train, x_test, y_train, y_test = set_timesteps.create_timeseries_data(train_x, test_x, train_y, test_y, timesteps)
def create_timeseries_data(train_x, test_x, train_y, test_y, timesteps):

    def reshape_data(data_x, timesteps):
        data_x_reshaped = []
        for i in range(timesteps, data_x.shape[0]):
            xset = []
            for j in range(data_x.shape[1]):
                d = data_x[i-timesteps:i, j]
                xset.append(d)
            xarr = np.array(xset).reshape(timesteps, data_x.shape[1])
            data_x_reshaped.append(xarr)
        return np.array(data_x_reshaped)

    ## ["データ数", "timesteps", "特徴量数"]の三次元構造に変換
    x_train = reshape_data(train_x, timesteps)
    x_test = reshape_data(test_x, timesteps)
    
    ## timesteps時間以降のデータを目的変数に設定
    y_train = train_y[timesteps:]
    y_test = test_y[timesteps:]

    return x_train, x_test, y_train, y_test

