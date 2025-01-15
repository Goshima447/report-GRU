#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import glob
import os

# 使用例：
# import load_dataset_1week
# dataset = load_load_dataset_1week.load_dataset_minute(5161)
# dataset = load_load_dataset_1week.load_dataset_hour(5161)


# 単一のデータセット読み込み（"CellID"を引数に指定する）
# 10分単位で測定したデータセットでのみ使用
def load_dataset_minute(file_id):
    # ファイルの読み込み（整数IDを含むファイル名）
    dataset = pd.read_csv(rf"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\Datasets_week\minute\mi-ID-{file_id}.csv")
    
    # 各列の型変換を辞書に格納
    conversion_dict = {
        'datetime': 'datetime64[ns]',
        'CellID': 'int16',
        'countrycode': 'int16',
        'smsin': 'float64',
        'smsout': 'float64',
        'callin': 'float64',
        'callout': 'float64',
        'internet': 'float64',
    }

    # 欠損値を0で補完
    dataset = dataset.fillna(0)
    
    # 型変換を適用
    dataset = dataset.astype(conversion_dict)
    
    # "smsin"及び"smsout"を"sms"として合計、"callin"及び"callout"を"call"として合計（行ごとに合計）
    dataset['sms'] = dataset['smsin'] + dataset['smsout']
    dataset['call'] = dataset['callin'] + dataset['callout']

    # 追加後のデータセットの整列（datetimeは除外）
    columns_order = ['datetime', 'CellID', 'sms', 'call', 'internet']
    dataset = dataset[columns_order]

    # "datetime"及び"CellID"にてグループ化し、値を合計
    dataset = dataset[['datetime', 'CellID', 'sms', 'call', 'internet']].groupby(['datetime', 'CellID'], as_index=False).sum()

    # 日、時間、分、曜日を取り出す
    dataset['day'] = dataset['datetime'].dt.day.astype('int8')
    dataset['hour'] = dataset['datetime'].dt.hour.astype('int8')
    dataset['minute'] = dataset['datetime'].dt.minute.astype('int8')
    dataset['weekday'] = dataset['datetime'].dt.dayofweek.astype('int8')   

    # 月の日数を計算（各月の最終日を取得）
    dataset['days_in_month'] = dataset['datetime'].dt.days_in_month

    # サインとコサイン変換を行う関数
    def encode(dataset, columns):
        for column in columns:
            if column == 'day':
                # 日に関しては、1か月の周期に基づく変換
                dataset[column + '_cos'] = np.cos(2 * np.pi * dataset[column] / dataset['days_in_month'])
                dataset[column + '_sin'] = np.sin(2 * np.pi * dataset[column] / dataset['days_in_month'])
            elif column == 'hour':
                # 時間に関しては、24時間周期に基づく変換
                dataset[column + '_cos'] = np.cos(2 * np.pi * dataset[column] / 24)
                dataset[column + '_sin'] = np.sin(2 * np.pi * dataset[column] / 24)
            elif column == 'minute':
                # 分に関しては、60分周期に基づく変換
                dataset[column + '_cos'] = np.cos(2 * np.pi * dataset[column] / 60)
                dataset[column + '_sin'] = np.sin(2 * np.pi * dataset[column] / 60)
            elif column == 'weekday':
                # 曜日に関しては、7日周期に基づく変換
                dataset[column + '_cos'] = np.cos(2 * np.pi * dataset[column] / 7)
                dataset[column + '_sin'] = np.sin(2 * np.pi * dataset[column] / 7)
        return dataset
    
    # サインとコサイン変換を'weekday', 'hour', 'minute', 'day'に適用
    dataset = encode(dataset, ['day', 'hour', 'minute', 'weekday'])
    
    ## 追加後のデータセットの整列
    columns_order = [
        'datetime', 'day_cos','day_sin', 'hour_cos','hour_sin', 'minute_cos', 'minute_sin', 'weekday_cos','weekday_sin',
        'sms', 'call', 'internet'
    ]
    dataset = dataset[columns_order]

    return dataset  # pandas Dataframe型で返す


## すべてのグリッドを読み込む場合
def load_dataset_hour(cell_id):
    ##　ディレクトリ内のcsvファイル読み込み （sms-call-internet-miの全ファイル）
    file_paths = glob.glob(os.path.join(r"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\Datasets_week\hour", "sms-call-internet-mi-*.csv"))

    ## 取得したファイルパスを順に読み込み、辞書に格納する（dfs["df_Day_1"], dfs["df_Day_2"],...の形式）
    dfs = {}
    for i, file_path in enumerate(file_paths, start=1):
        dfs[f"df_Day_{i}"] = pd.read_csv(file_path)

    ## 各列の型変換を辞書に格納
    conversion_dict = {
        'datetime': 'datetime64[ns]',
        'CellID': 'int16',
        'countrycode': 'int16',
        'smsin': 'float32',
        'smsout': 'float32',
        'callin': 'float32',
        'callout': 'float32',
        'internet': 'float32',
    }

    ## 各データセットの欠損値を変換（0としている）
    for key in dfs:
        dfs[key].fillna(0, inplace=True)

    ## 型変換とデータセット連結
    dataset = pd.DataFrame()
    for key in dfs:
        dataset = pd.concat([dataset, dfs[key].astype(conversion_dict)], axis=0)  # 行方向に連結

    ## "smsin"及び"smsout"を"sms"として合計、"callin"及び"callout"を"call"として合計（行ごとに合計）
    dataset['sms'] = dataset['smsin'] + dataset['smsout']
    dataset['call'] = dataset['callin'] + dataset['callout']

    ## 追加後のデータセットの整列（datetimeは除外）
    columns_order = ['datetime', 'CellID', 'sms', 'call', 'internet']
    dataset = dataset[columns_order]

    ## "datetime"及び"CellID"にてグループ化し、値を合計
    dataset = dataset[['datetime', 'CellID', 'sms', 'call', 'internet']].groupby(['datetime', 'CellID'], as_index=False).sum()

    ## 日、時間、曜日を取り出す
    dataset['day'] = dataset['datetime'].dt.day.astype('int8')
    dataset['hour'] = dataset['datetime'].dt.hour.astype('int8')
    dataset['weekday'] = dataset['datetime'].dt.dayofweek.astype('int8')
    

    # 月の日数を計算（各月の最終日を取得）
    dataset['days_in_month'] = dataset['datetime'].dt.days_in_month

    # サインとコサイン変換を行う関数
    def encode(dataset, columns):
        for column in columns:
            if column == 'day':
                # 日に関しては、1か月の周期に基づく変換
                dataset[column + '_cos'] = np.cos(2 * np.pi * dataset[column] / dataset['days_in_month'])
                dataset[column + '_sin'] = np.sin(2 * np.pi * dataset[column] / dataset['days_in_month'])
            elif column == 'hour':
                # 時間に関しては、24時間周期に基づく変換
                dataset[column + '_cos'] = np.cos(2 * np.pi * dataset[column] / 24)
                dataset[column + '_sin'] = np.sin(2 * np.pi * dataset[column] / 24)
            elif column == 'minute':
                # 分に関しては、60分周期に基づく変換
                dataset[column + '_cos'] = np.cos(2 * np.pi * dataset[column] / 60)
                dataset[column + '_sin'] = np.sin(2 * np.pi * dataset[column] / 60)
            elif column == 'weekday':
                # 曜日に関しては、7日周期に基づく変換
                dataset[column + '_cos'] = np.cos(2 * np.pi * dataset[column] / 7)
                dataset[column + '_sin'] = np.sin(2 * np.pi * dataset[column] / 7)
        return dataset
    
    # サインとコサイン変換を'weekday', 'hour', 'day'に適用
    dataset = encode(dataset, ['day', 'hour', 'weekday'])
    
    # 指定したIDのデータを取得
    dataset = dataset[dataset['CellID'].isin(cell_id)]
    columns_order = ['datetime', 'day_cos','day_sin', 'hour_cos','hour_sin', 'weekday_cos','weekday_sin', 'sms', 'call', 'internet']
    dataset = dataset[columns_order]

    return dataset  # pandas Dataframe型で返す


