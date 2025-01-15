#!/usr/bin/env python
# coding: utf-8


import pandas as pd

# 指定したCell-IDのデータセットを取得
# 引数cell_idsは対応する数字をリストとして受け取る

# 使用例：
#import get_sms_call_internet_by_ID
#dataset = get_sms_call_internet_by_ID.get_by_ID(dataset, [1]) 

def get_by_ID(dataset, cell_ids):
    dataset = dataset[dataset['CellID'].isin(cell_ids)]
    columns_order = ['datetime', 'day_cos','day_sin', 'hour_cos','hour_sin', 'weekday_cos','weekday_sin', 'sms', 'call', 'internet']
    dataset = dataset[columns_order]
    return dataset


