#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import datetime
import glob
import os


# time_unitには時間単位を指定（'hour'or'minute'）, file_IDにはファイル名 mi-ID- 以降の整数を入力
# 使用例： 
# import load_sms_call_internet_mi
# train, test, y_datetime = load_sms_call_internet_mi.load_dataset_ID('minute', 1)
def load_dataset_ID(time_unit, file_id):
    # 各列の型変換を辞書に格納
    conversion_dict = {
        'datetime': 'datetime64[ns]',
        'CellID': 'int16',
        'sms': 'float64',
        'call': 'float64',
        'internet': 'float64',
    }

    # サインとコサイン変換を行う関数
    def encode(dataset, columns):
        # 月の日数を計算（各月の最終日を取得）
        dataset['days_in_month'] = dataset['datetime'].dt.days_in_month
            
        # columns に基づいて各カラムを計算
        for column in columns:
            if column == 'day':
                dataset['day'] = dataset['datetime'].dt.day.astype('int8')
                dataset[column + '_cos'] = np.cos(2 * np.pi * dataset[column] / dataset['days_in_month'])
                dataset[column + '_sin'] = np.sin(2 * np.pi * dataset[column] / dataset['days_in_month'])
            elif column == 'hour':
                dataset['hour'] = dataset['datetime'].dt.hour.astype('int8')
                dataset[column + '_cos'] = np.cos(2 * np.pi * dataset[column] / 24)
                dataset[column + '_sin'] = np.sin(2 * np.pi * dataset[column] / 24)
            elif column == 'minute':
                dataset['minute'] = dataset['datetime'].dt.minute.astype('int8')
                dataset[column + '_cos'] = np.cos(2 * np.pi * dataset[column] / 60)
                dataset[column + '_sin'] = np.sin(2 * np.pi * dataset[column] / 60)
            elif column == 'weekday':
                dataset['weekday'] = dataset['datetime'].dt.dayofweek.astype('int8')
                dataset[column + '_cos'] = np.cos(2 * np.pi * dataset[column] / 7)
                dataset[column + '_sin'] = np.sin(2 * np.pi * dataset[column] / 7)
                
        return dataset
    
    
    # time_unitにて指定された集計時間に応じて処理内容を変更する
    # 1時間間隔の場合
    if time_unit == 'hour':
        # Datasets_hour内の訓練データとテストデータ及び、テストデータの日付情報を取得
        train_dataset = pd.read_csv(rf"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\Datasets_hour\train\train_mi-ID-{file_id}.csv")
        test_dataset = pd.read_csv(rf"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\Datasets_hour\test\test_mi-ID-{file_id}.csv")
        y_datetime = test_dataset['datetime']
        
        # 型変換を適用
        train_dataset = train_dataset.astype(conversion_dict)
        test_dataset = test_dataset.astype(conversion_dict)
        
        # sin,cos変換を'hour', 'day', 'weekday'に適用
        train_dataset = encode(train_dataset, ['day', 'hour', 'weekday'])
        test_dataset = encode(test_dataset, ['day', 'hour', 'weekday'])
        
        # 追加後のデータセットの整列
        columns_order = ['day_cos','day_sin', 'hour_cos','hour_sin', 'weekday_cos','weekday_sin', 'sms', 'call', 'internet']
        train_dataset = train_dataset[columns_order]
        test_dataset = test_dataset[columns_order]

    # 10分間隔の場合
    elif time_unit == 'minute':
        # Datasets_minute内の訓練データとテストデータ及び、テストデータの日付情報を取得
        train_dataset = pd.read_csv(rf"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\Datasets_minute\train\train_mi-ID-{file_id}.csv")
        test_dataset = pd.read_csv(rf"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\Datasets_minute\test\test_mi-ID-{file_id}.csv")
        y_datetime = test_dataset['datetime']

        # 型変換を適用
        train_dataset = train_dataset.astype(conversion_dict)
        test_dataset = test_dataset.astype(conversion_dict)

        # sin,cos変換を'hour', 'day', 'minute', 'weekday'に適用
        train_dataset = encode(train_dataset, ['day', 'hour', 'minute', 'weekday'])
        test_dataset = encode(test_dataset, ['day', 'hour', 'minute', 'weekday'])

        # 追加後のデータセットの整列
        columns_order = [
            'day_cos','day_sin', 'hour_cos','hour_sin', 'minute_cos', 'minute_sin', 'weekday_cos','weekday_sin',
            'sms', 'call', 'internet'
        ]
        train_dataset = train_dataset[columns_order]
        test_dataset = test_dataset[columns_order]

    # 1時間間隔 & Unixtime
    elif time_unit == 'hour_unix':
        # Datasets_hour内の訓練データとテストデータ及び、テストデータの日付情報を取得
        train_dataset = pd.read_csv(rf"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\Datasets_hour\train\train_mi-ID-{file_id}.csv")
        test_dataset = pd.read_csv(rf"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\Datasets_hour\test\test_mi-ID-{file_id}.csv")
        y_datetime = test_dataset['datetime']
        
        # 型変換を適用
        train_dataset = train_dataset.astype(conversion_dict)
        test_dataset = test_dataset.astype(conversion_dict)

        # datetime列をUnix-timestampに変更（ナノ秒単位に変換後、ミリ秒単位に変換）
        train_dataset['datetime'] = train_dataset['datetime'].astype(np.int64) // 10**6 
        test_dataset['datetime'] = test_dataset['datetime'].astype(np.int64) // 10**6 
        
        # 変更後のデータセットの整列
        columns_order = ['datetime', 'sms', 'call', 'internet']
        train_dataset = train_dataset[columns_order]
        test_dataset = test_dataset[columns_order]

    # 10分間隔 & Unixtime
    elif time_unit == 'minute_unix':
        # Datasets_minute内の訓練データとテストデータ及び、テストデータの日付情報を取得
        train_dataset = pd.read_csv(rf"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\Datasets_minute\train\train_mi-ID-{file_id}.csv")
        test_dataset = pd.read_csv(rf"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\Datasets_minute\test\test_mi-ID-{file_id}.csv")
        y_datetime = test_dataset['datetime']

        # 型変換を適用
        train_dataset = train_dataset.astype(conversion_dict)
        test_dataset = test_dataset.astype(conversion_dict)

        # datetime列をUnix-timestampに変更（ナノ秒単位に変換後、ミリ秒単位に変換）
        train_dataset['datetime'] = train_dataset['datetime'].astype(np.int64) // 10**6 
        test_dataset['datetime'] = test_dataset['datetime'].astype(np.int64) // 10**6 
        
        # 変更後のデータセットの整列
        columns_order = ['datetime', 'sms', 'call', 'internet']
        train_dataset = train_dataset[columns_order]
        test_dataset = test_dataset[columns_order]

    # エラー処理
    else:
        raise ValueError("time_unit must be 'hour', 'minute', 'hour_unix' or 'minute_unix'")
    
    
    return train_dataset, test_dataset, y_datetime  # pandas Dataframe型で返す




