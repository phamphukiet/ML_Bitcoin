import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
import csv

def robust_future_predict(model, df, scaler_x, scaler_y, pre_day, predict_days=5, use_true_ratio=0.5):
    FEATURE_COLS = ['H-L', 'O-C', 'SMA_7', 'SMA_14', 'SMA_21', 'SD_7', 'SD_21']
    scaled_x = scaler_x.transform(df[FEATURE_COLS])
    last_window = scaled_x[-pre_day:]
    preds_future = []
    current = last_window.copy()

    for i in range(predict_days):
        pred = model.predict(current.reshape(1, pre_day, current.shape[1]))
        preds_future.append(pred[0,0])
        new_row = current[-1].copy()

        # Nếu còn dữ liệu thật thì lấy ra trộn
        if i < len(df):
            true_val = scaler_y.transform(df[['Close']].iloc[[-(i+1)]]).ravel()[0]
            new_row[-1] = use_true_ratio * true_val + (1 - use_true_ratio) * pred
        else:
            new_row[-1] = pred

        current = np.vstack([current[1:], new_row])

    preds_future = scaler_y.inverse_transform(np.array(preds_future).reshape(-1,1))
    return preds_future

def ensemble_predict(model_entries, df, pre_day, predict_days=5):
    preds_all = []
    for entry in model_entries:
        model = load_model(entry["model_path"])
        scaler_x = joblib.load(entry["scaler_x"])
        scaler_y = joblib.load(entry["scaler_y"])

        # chuẩn hóa dữ liệu
        FEATURE_COLS = ['H-L', 'O-C', 'SMA_7', 'SMA_14', 'SMA_21', 'SD_7', 'SD_21']
        scaled_x = scaler_x.transform(df[FEATURE_COLS])
        last_window = scaled_x[-pre_day:]

        preds_future = []
        current = last_window.copy()
        for _ in range(predict_days):
            pred = model.predict(current.reshape(1, pre_day, len(FEATURE_COLS)))
            preds_future.append(pred[0, 0])
            new_row = current[-1].copy()
            new_row[-1] = pred
            current = np.vstack([current[1:], new_row])

        preds_future = scaler_y.inverse_transform(np.array(preds_future).reshape(-1, 1))
        preds_all.append(preds_future)

    # Lấy trung bình của tất cả model
    preds_mean = np.mean(np.array(preds_all), axis=0)
    return preds_mean

def save_future_forecast(preds_future, future_dates, version, out_dir="models"):
    path = os.path.join(out_dir, "future_forecast.csv")
    file_exists = os.path.isfile(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["version", "date", "predicted_close"])
        for d, p in zip(future_dates, preds_future):
            writer.writerow([version, d.strftime("%Y-%m-%d"), round(float(p[0]), 2)])

    print(f"📑 Đã ghi future forecast vào {path}")