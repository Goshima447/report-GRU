#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# 値を0～1の範囲に正規化し、pandas DataframeからNumPy配列に変更（mms_yは評価指標に使用）
# 使用例：
# import normalize_train_test
# train_x, test_x, train_y, test_y, mms_y = normalize_train_test.apply_MMS(train_x, test_x, train_y, test_y)
def apply_MMS(train_x, test_x, train_y, test_y):
    # 'sms', 'call'を正規化（データセットの末尾2列）
    # 'sms', 'call'の列が存在するかどうか判定する（smsとcallの列以外は"MinMaxScaler"に適していないため）
    if {'sms', 'call'}.issubset(train_x.columns) and {'sms', 'call'}.issubset(test_x.columns):
        mms_x = MinMaxScaler()
        mms_x.fit(train_x.iloc[:, -2:]) 
        train_x.iloc[:, -2:] = mms_x.transform(train_x.iloc[:, -2:])
        test_x.iloc[:, -2:] = mms_x.transform(test_x.iloc[:, -2:])
        #print("Columns 'sms' and 'call' are present in both.")
    else:
        print("Columns 'sms' and 'call' are missing in one or both.")
    
    # DataFrameからNumPy配列に変換
    train_x = train_x.to_numpy(dtype=np.float64)
    test_x = test_x.to_numpy(dtype=np.float64)

    # 目的変数を一次元配列から二次元配列に変換
    mms_y = MinMaxScaler()
    mms_y.fit(train_y.to_numpy().reshape(-1, 1))
    train_y = mms_y.transform(train_y.to_numpy().reshape(-1, 1))
    test_y = mms_y.transform(test_y.to_numpy().reshape(-1, 1))

    # 正規化した目的変数（mms_y）は結果表示で用いる
    return train_x, test_x, train_y, test_y, mms_y

