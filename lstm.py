from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from itertools import product
from time import perf_counter
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def read_table(path, sheet):
    return pd.read_excel(path, sheet_name=sheet)

def load_series(path, date_col, target_col, sheet=None):
    df = read_table(path, sheet)
    df.columns = [str(c).strip() for c in df.columns]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    s = pd.to_numeric(df[target_col], errors='coerce')
    return s.dropna()

def split_by_ratio(s, train_size):
    n = int(len(s) * train_size)
    return (s.iloc[:n], s.iloc[n:])

def make_windows(arr, lookback):
    X, y = ([], [])
    for i in range(len(arr) - lookback):
        X.append(arr[i:i + lookback])
        y.append(arr[i + lookback])
    X = np.array(X).reshape((-1, lookback, 1))
    y = np.array(y).reshape((-1, 1))
    return (X, y)

def build_model(lookback):
    model = Sequential([
        LSTM(100, activation='tanh', recurrent_activation='sigmoid',
             return_sequences=True, use_bias=True, dropout=0.0, recurrent_dropout=0.0,
             input_shape=(lookback, 1)),
        LSTM(100, activation='tanh', recurrent_activation='sigmoid',
             return_sequences=False, use_bias=True, dropout=0.0, recurrent_dropout=0.0),
        Dense(100, activation=None, use_bias=False,
              kernel_initializer=None, bias_initializer=None,
              kernel_regularizer=None, activity_regularizer=None,
              kernel_constraint=None, bias_constraint=None),
        Dense(1,   activation=None, use_bias=False,
              kernel_initializer=None, bias_initializer=None,
              kernel_regularizer=None, activity_regularizer=None,
              kernel_constraint=None, bias_constraint=None),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def try_combo(series, lookback, train_size, epochs, batch):
    scaler = MinMaxScaler()
    train_s, val_s = split_by_ratio(series, train_size)
    tr = scaler.fit_transform(train_s.values.reshape(-1, 1)).ravel()
    va = scaler.transform(val_s.values.reshape(-1, 1)).ravel()
    tail = tr[-lookback:] if len(tr) >= lookback else tr
    X_tr, y_tr = make_windows(tr, lookback)
    X_va, y_va = make_windows(np.concatenate([tail, va]), lookback)
    if len(X_tr) == 0 or len(X_va) == 0:
        return np.inf
    model = build_model(lookback)
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    t0 = perf_counter()
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs, batch_size=batch, verbose=1, callbacks=[es])
    train_time = perf_counter() - t0
    pred_va_scaled = model.predict(X_va, verbose=0).ravel()
    inv = lambda a: scaler.inverse_transform(a.reshape(-1, 1)).ravel()
    y_true = inv(y_va.ravel())
    y_pred = inv(pred_va_scaled)
    y_pred = np.maximum(y_pred, 0.0)
    mse = mean_squared_error(y_true, y_pred)
    print(f'[try_combo] lb={lookback} ts={train_size} ep={epochs} ba={batch} mse={mse:.4f} time={train_time:.1f}s')
    return mse

def run_grid(series, lookback_opts, train_size_opts, epoch_opts, batch_opts):
    combos = list(product(lookback_opts, train_size_opts, epoch_opts, batch_opts))
    best = None
    t0_total = perf_counter()
    for lb, ts, ep, ba in combos:
        t0 = perf_counter()
        mse = try_combo(series, lb, ts, ep, ba)
        dt = perf_counter() - t0
        if np.isfinite(mse) and (best is None or mse < best[0]):
            best = (mse, {'lookback': lb, 'train_size': ts, 'epochs': ep, 'batch': ba})
        print(f'Grid combo lb={lb} ts={ts} ep={ep} ba={ba} -> mse={mse:.4f} took={dt:.1f}s | best={best}')
    total = perf_counter() - t0_total
    print(f'Grid search finished in {total:.1f}s. Best: {best}')
    return best

def train_final(series, lookback, train_size, epochs, batch, title, out):
    out.mkdir(parents=True, exist_ok=True)
    scaler = MinMaxScaler()
    train_s, val_s = split_by_ratio(series, train_size)
    tr = scaler.fit_transform(train_s.values.reshape(-1, 1)).ravel()
    va = scaler.transform(val_s.values.reshape(-1, 1)).ravel()
    X_tr, y_tr = make_windows(tr, lookback)
    tail = tr[-lookback:] if len(tr) >= lookback else tr
    X_va, y_va = make_windows(np.concatenate([tail, va]), lookback)
    model = build_model(lookback)
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    t0 = perf_counter()
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs, batch_size=batch, verbose=1, callbacks=[es])
    train_time = perf_counter() - t0
    pred_va_scaled = model.predict(X_va, verbose=0).ravel()
    inv = lambda a: scaler.inverse_transform(a.reshape(-1, 1)).ravel()
    y_true = inv(y_va.ravel())
    y_pred = inv(pred_va_scaled)
    y_pred = np.maximum(y_pred, 0.0)
    model.save(out / 'model.keras')
    joblib.dump(scaler, out / 'scaler.joblib')
    start = max(0, lookback - len(tr))
    n = min(len(y_pred), len(y_true), len(val_s.index) - start)
    if n <= 0:
        raise ValueError(f'Nothing to align: len(tr)={len(tr)}, len(val)={len(val_s)}, lookback={lookback}, start={start}, len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}')
    dates_aligned = val_s.index[start:start + n]
    y_true_aligned = y_true[:n]
    y_pred_aligned = y_pred[:n]
    pd.DataFrame({'date': dates_aligned, 'y_true': y_true_aligned, 'y_pred': y_pred_aligned}).to_csv(out / 'predictions.csv', index=False)
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Energy')
    plt.plot(train_s.index, train_s.values, label='Train')
    plt.plot(dates_aligned, y_true_aligned, label='Validation')
    plt.plot(dates_aligned, y_pred_aligned, label='Pred')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / 'predictions.png', dpi=150)
    plt.close()
    mse = mean_squared_error(y_true_aligned, y_pred_aligned)
    print('final ->', f'lookback={lookback} train_size={train_size} epochs={epochs} batch={batch} mse={mse:.6f}')
    print(f'training_time_sec={train_time:.1f}')
    print('saved to:', out.resolve())
