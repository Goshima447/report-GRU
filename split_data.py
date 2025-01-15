#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


## データを 8:2 の割合で分割

## 使用例：
# feature = dataset.drop(columns=['internet'])
# target = dataset[['datetime', 'internet']]
# import split_data
# train_x, test_x, train_y, test_y, y_datetime = split_data.split_data_8_2(feature, target)


def split_data_8_2(feature, target):
    train_x, test_x, train_y, test_y = train_test_split(feature, target, test_size=0.2, random_state=1, shuffle=False)

    ## "datetime"の情報を取り出し、除外する
    y_datetime = test_y[['datetime']]
    train_y = train_y.drop(columns=['datetime'])
    test_y = test_y.drop(columns=['datetime'])
    
    return tuple(df.reset_index(drop=True) for df in [train_x, test_x, train_y, test_y, y_datetime])


