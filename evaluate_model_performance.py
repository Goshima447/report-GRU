#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score


# テストデータに対する予測を行い、評価結果を表示
# 使用例： 
# import evaluate_model_performance
# y_test_pred, y_test_true = evaluate_model_performance.evaluate_model(x_test, y_test, mms_y, model, seasonal_lag)

def evaluate_model(x_test, y_test, scaler, model, seasonal_lag=1):
    ## MdAPE (Median Absolute Percentage Error)
    def calculate_mdape(y_true, y_pred):
        # 実測値がゼロの部分を除外
        mask = y_true != 0
        ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
        return np.median(ape)

    ## GMAPE (Geometric Mean Absolute Percentage Error)
    def calculate_gmape(y_true, y_pred):
        # 実測値がゼロの部分を除外
        mask = y_true != 0
        ape_plus_one = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) + 1
        gmape = np.prod(ape_plus_one) ** (1 / len(ape_plus_one)) - 1
        return gmape * 100

    ## MASE（Mean Absolute Scaled Error）
    def mase(y_true, y_pred, seasonal_lag):
        naive_error = np.mean(np.abs(y_true[seasonal_lag:] - y_true[:-seasonal_lag]))
        scaled_error = np.mean(np.abs(y_true - y_pred)) / naive_error
        return scaled_error

    ## SMAE（Scaled Mean Absolute Error）
    def smae(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        return mae / np.mean(y_true)
    
    ## テストデータに対する予測
    y_test_pred = model.predict(x_test)

    ## 予測結果および "y_test" を元のスケールに変換
    y_test_pred = scaler.inverse_transform(y_test_pred)
    y_test_true = scaler.inverse_transform(y_test)
    
    ## 評価指標を計算
    RMSE = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    MAE = mean_absolute_error(y_test_true, y_test_pred)
    MAPE = mean_absolute_percentage_error(y_test_true, y_test_pred)
    MdAPE = calculate_mdape(y_test_true, y_test_pred)
    R2 = r2_score(y_test_true, y_test_pred)
    MASE = mase(y_test_true, y_test_pred, seasonal_lag)
    

    # 指標出力
    print('RMSE:', RMSE)
    print('MAE:', MAE)
    print('MAPE:', MAPE)
    print('MdAPE:', MdAPE)
    print('R2:', R2)
    print('MASE:', MASE)

    return y_test_pred, y_test_true


