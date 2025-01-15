#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter
from tensorflow.keras.models import load_model

from load_sms_call_internet_mi import load_dataset_ID
from normalize_train_test import apply_MMS
from set_timesteps import create_timeseries_data
from build_model import build_model_GRU
from evaluate_model_performance import evaluate_model



# time_unit, CellID及びtimestepsを受け取り、予測までを行う関数（CellIDを指定して実行）
# 使用例：
# from apply_best_model import analyze_data_traffic
# analyze_data_traffic(cell_id, 'hour', 24)
def analyze_data_traffic(CellID, time_unit, timesteps):
    # 不正な空白が含まれている可能性に備えてtime_unitを整形
    time_unit = time_unit.strip() 

    # CellIDを表示
    print(f"Analyzing data for CellID: {CellID}")
    
    # データセット読み込み
    train, test, y_datetime = load_dataset_ID(time_unit, CellID)

    # 説明変数と目的変数に分割
    train_x, train_y = train.drop(columns=['internet']), train['internet']
    test_x, test_y = test.drop(columns=['internet']), test['internet']

    # データの正規化
    train_x, test_x, train_y, test_y, mms_y = apply_MMS(train_x, test_x, train_y, test_y)

    # データの時系列変換
    x_train, x_test, y_train, y_test = create_timeseries_data(train_x, test_x, train_y, test_y, timesteps)

    # 最良モデルのロード
    model_path = os.path.join(r"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\best_model", time_unit, "best_model_minute.h5")
    model = load_model(model_path)


    # テストデータの予測
    y_test_pred, y_test_true = evaluate_model(x_test, y_test, mms_y, model)

    # 時系列データに合わせてプロット
    datetime = pd.to_datetime(y_datetime[timesteps:].reset_index(drop=True))
    plt.figure(figsize=(30, 8))
    plt.plot(datetime, y_test_true, linestyle='--', color='blue', label='True')
    plt.plot(datetime, y_test_pred, color='orange', label='Predicted')
    plt.gca().xaxis.set_major_locator(AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.ylabel('Value')
    plt.xlabel(f'Datetime (CellID: {CellID})') 
    plt.legend(loc='upper right')
    plt.ylim(0, 50000)
    plt.show()
    

    # 出力ディレクトリとファイルパス設定
    output_dir = os.path.join(
        r"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\Datasets_output",
        time_unit,
    )
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{time_unit}-mi-ID-{CellID}.csv")

    # 予測結果の保存
    dataset = pd.DataFrame({
        "datetime": datetime,
        "y_test_true": np.ravel(y_test_true),
        "y_test_pred": np.ravel(y_test_pred)
    })
    dataset.to_csv(file_path, index=False, encoding="utf-8")