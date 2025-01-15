#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from get_cellid_within_range import get_row_col, get_cells_in_range

# セルIDに対応するファイルを読み込む関数。返り値はファイルパスの辞書であり、値は y_test_pred（3列目）の値が格納
#from load_output_internet import load_output
#files_data = load_output(5161, 3, 'hour')
def load_output(center_cell_id, range_size, time_unit):

    if time_unit not in ['hour', 'minute']:
        raise ValueError("時間単位は 'hour' または 'minute' を指定してください。")
    
    # ベースパスを設定（'hour' または 'minute' ディレクトリに基づいて）
    base_path = r"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\Datasets_output"
    dir_path = os.path.join(base_path, time_unit)
    
    # セルID範囲を取得
    cells_in_range = get_cells_in_range(center_cell_id, range_size)
    
    # データを格納するリスト
    files_data = []
    
    # セルIDに対応するファイルを読み込む
    for cell_id in cells_in_range:
        # ファイル名を作成: "hour-mi-ID-<cell_id>.csv" または "minute-mi-ID-<cell_id>.csv"
        file_name = f"{time_unit}-mi-ID-{cell_id}.csv"
        file_path = os.path.join(dir_path, file_name)

        # ファイルが存在するか確認し、読み込む
        if os.path.exists(file_path):
            # CSVファイルを読み込む（3列目のみを読み込む）
            df = pd.read_csv(file_path, usecols=[2])  # （インデックスは0から始まるため2）
            df[df < 0] = 0.001  # 負の値があった場合は、1に変換
            files_data.append(df.iloc[:, 0].tolist())  # 全データをリストに追加
        else:
            print(f"ファイルが存在しません: {file_path}")
    
    return files_data

