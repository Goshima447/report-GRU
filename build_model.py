#!/usr/bin/env python
# coding: utf-8


import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import GlorotUniform

from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback



# ランダムシードを設定
seed = 1
random.seed(seed)               # Pythonの標準シード
np.random.seed(seed)            # NumPyのシード
tf.random.set_seed(seed)        # TensorFlowのシード

# GPU0のみを使用するように設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# cuDNNの非決定論的動作を無効化する設定
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# 決定論的な動作を有効化（結果を再現可能にする）
tf.config.experimental.enable_op_determinism()

# メモリの動的割り当て
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 各GPUでメモリの動的割り当てを有効化
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled dynamic memory allocation.")
    except RuntimeError as e:
        print(f"Error enabling dynamic memory allocation: {e}")
else:
    print("No GPUs detected.")


## ベイズ最適化にてパラメータを最適化するGRUモデルの構築
## 使用例： model, history = build_model.build_model_GRU_Optuna(x_train, y_train, timesteps, time_unit)
def build_model_GRU_Optuna(x_train, y_train, timesteps, time_unit):
    # 訓練データと検証データを時系列順に分割
    train_size = int(len(x_train) * 0.8)
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]
    
    def objective(trial):
        # セッションをクリア
        tf.keras.backend.clear_session()
    
        # ハイパーパラメータの探索範囲
        neuron = trial.suggest_categorical('neuron', [32, 64, 128, 256])
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    
        # ドロップアウトは0.2に固定
        dropout_rate = 0.2
    
        # GPUでモデルを構築・学習
        with tf.device('/GPU:0'):
            model_GRU = Sequential()
            model_GRU.add(GRU(
                neuron,
                activation="tanh", 
                batch_input_shape=(None, timesteps, x_train.shape[2]),
                return_sequences=False,
                kernel_initializer=GlorotUniform(seed=1)
            ))
            model_GRU.add(Dropout(0.2))  # ここでドロップアウトを0.2に固定
            model_GRU.add(Dense(1, activation="linear"))
    
            optimizer = Adam(learning_rate=learning_rate)
            model_GRU.compile(loss="mean_squared_error", optimizer=optimizer)
    
            def report_epoch_end(epoch, logs):
                trial.report(logs['val_loss'], step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
    
            callback = LambdaCallback(on_epoch_end=report_epoch_end)
            
            # EarlyStoppingのコールバックを追加
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                min_delta=0.0001,
                restore_best_weights=True  # 最良の重みを復元
            )
    
            history = model_GRU.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=100,
                batch_size=batch_size,
                shuffle=False,
                verbose=0,
                callbacks=[callback, early_stopping]  # ここでEarlyStoppingを追加
            )
        
        val_loss = history.history['val_loss'][-1]
        return val_loss

    
    # MedianPrunerを使用したStudyの作成
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=1), pruner=pruner)
    study.optimize(objective, n_trials=100, n_jobs=1) # 試行回数は100回

    # 最適なハイパーパラメータを取得
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    
    # 最適なハイパーパラメータでモデルを再構築
    tf.keras.backend.clear_session()
    with tf.device('/GPU:0'):
        best_model = Sequential()
        best_model.add(GRU(
            best_params['neuron'],
            activation="tanh", 
            batch_input_shape=(None, timesteps, x_train.shape[2]),
            return_sequences=False,
            kernel_initializer=GlorotUniform(seed=1)
        ))
        best_model.add(Dropout(0.2))  # ここでドロップアウトを0.2に固定
        best_model.add(Dense(1, activation="linear"))

        optimizer = Adam(learning_rate=best_params['learning_rate'])
        best_model.compile(loss="mean_squared_error", optimizer=optimizer)

        # EarlyStoppingを再度追加
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.0001,
            restore_best_weights=True
        )

        history_GRU = best_model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=100,
            batch_size=best_params['batch_size'],
            shuffle=False,
            verbose=0,
            callbacks=[early_stopping]  # 最良のモデルの重みを保持
        )
        
        # time_unitに基づいて保存ディレクトリを決定
        save_dir = os.path.join(
            r"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\best_model",
            time_unit,  # 'hour'または'minute'に応じてディレクトリを作成
        )
        
        # 保存先ディレクトリを作成（存在しない場合）
        os.makedirs(save_dir, exist_ok=True)
        
        # モデルを保存
        model_path = os.path.join(save_dir, "best_model.h5")
        best_model.save(model_path)
        print(f"Best model saved as '{model_path}'")
        
    return best_model, history_GRU

# モデルを2層に増やした場合
#import build_model
#model, history = build_model.build_model_GRU_Optuna_layers(x_train, y_train, timesteps, time_unit)
def build_model_GRU_Optuna_layers(x_train, y_train, timesteps, time_unit):
    
    # 訓練データと検証データを時系列順に分割
    train_size = int(len(x_train) * 0.8)
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]
    
    # Optunaで最適化を行う目的関数
    def objective(trial):
        # セッションをクリアして、メモリリークを防ぐ
        tf.keras.backend.clear_session()

        # 各ハイパーパラメータの探索範囲を設定
        neuron_1 = trial.suggest_categorical('neuron_1', [32, 64, 128])  # 1層目のユニット数
        neuron_2 = trial.suggest_categorical('neuron_2', [16, 32, 64])   # 2層目のユニット数
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)

        # モデル構築
        with tf.device('/GPU:0'):  # GPUを指定
            # モデルのインスタンス化
            model_GRU = Sequential()
            
            # 1層目のGRU
            model_GRU.add(GRU(
                neuron_1,
                activation="tanh", 
                batch_input_shape=(None, timesteps, x_train.shape[2]),
                return_sequences=True,
                kernel_initializer=GlorotUniform(seed=1)  # 重み初期化にシードを設定
            ))
            model_GRU.add(Dropout(0.2))  # ドロップアウト率を直接指定
            
            # 2層目のGRU
            model_GRU.add(GRU(
                neuron_2,
                activation="tanh",
                return_sequences=False,
                kernel_initializer=GlorotUniform(seed=1)
            ))
            model_GRU.add(Dropout(0.2))  # ドロップアウト率を直接指定
            
            model_GRU.add(Dense(1, activation="linear"))

            # Adamオプティマイザーのインスタンス化
            optimizer = Adam(learning_rate=learning_rate) 
            model_GRU.compile(loss="mean_squared_error", optimizer=optimizer)

            # EarlyStoppingを追加
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                min_delta=0.0001,
                restore_best_weights=True
            )
            
            # モデルのトレーニング
            history = model_GRU.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=100,
                batch_size=batch_size,
                shuffle=False,
                verbose=0,
                callbacks=[early_stopping]
            )
        
        # 最終エポックのバリデーション損失を返す
        val_loss = history.history['val_loss'][-1]
        trial.report(val_loss, step=100)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return val_loss

    # MedianPrunerを使用したStudyの作成
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=1), pruner=pruner)
    study.optimize(objective, n_trials=100, n_jobs=1)  # 試行回数を50回に設定
    
    # 最適なハイパーパラメータの取得
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    
    # 最適なモデルの作成前にセッションをクリア
    tf.keras.backend.clear_session() 
    
    # 最適なハイパーパラメータでモデルを再構築
    with tf.device('/GPU:0'):  # GPUを指定
        best_model = Sequential()
        
        # 1層目のGRU
        best_model.add(GRU(
            best_params['neuron_1'],
            activation="tanh", 
            batch_input_shape=(None, timesteps, x_train.shape[2]),
            return_sequences=True,
            kernel_initializer=GlorotUniform(seed=1)
        ))
        best_model.add(Dropout(0.2))  # ドロップアウト率を直接指定

        # 2層目のGRU
        best_model.add(GRU(
            best_params['neuron_2'],
            activation="tanh", 
            return_sequences=False,
            kernel_initializer=GlorotUniform(seed=1)
        ))
        best_model.add(Dropout(0.2))  # ドロップアウト率を直接指定
        
        best_model.add(Dense(1, activation="linear"))
        
        # 最適な学習率でオプティマイザーを設定
        optimizer = Adam(learning_rate=best_params['learning_rate']) 
        best_model.compile(loss="mean_squared_error", optimizer=optimizer)
        
        # EarlyStoppingを追加して学習
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.0001,
            restore_best_weights=True
        )
        
        # 最適なモデルで再度学習を行い、historyを取得
        history_GRU = best_model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=100,
            batch_size=best_params['batch_size'],
            shuffle=False,
            verbose=0,
            callbacks=[early_stopping]
        )
        
        # time_unitに基づいて保存ディレクトリを決定
        save_dir = os.path.join(
            r"C:\Users\goshima\Documents\卒業研究_AI\Python\Datasets\best_model",
            time_unit,  # 'hour'または'minute'に応じてディレクトリを作成
            'two_layer'
        )
        
        # 保存先ディレクトリを作成（存在しない場合）
        os.makedirs(save_dir, exist_ok=True)
        
        # モデルを保存
        model_path = os.path.join(save_dir, "best_model.h5")
        best_model.save(model_path)
        print(f"Best model saved as '{model_path}'")
    
    return best_model, history_GRU
