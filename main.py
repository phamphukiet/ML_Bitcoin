import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import (
    CRYPTO, START_DATE, END_DATE, PRE_DAY, TEST_RATIO,
    EPOCHS, BATCH_SIZE, MODELS_DIR, PREDICT_DAYS
)
from utils import data_utils, scaler_utils, model_utils, train_utils, plot_utils, analytic_utils

FEATURE_COLS = ['H-L', 'O-C', 'SMA_7', 'SMA_14', 'SMA_21', 'SD_7', 'SD_21']
TARGET_COL = 'Close'

def load_and_prepare_data():
    print(f"📥 Loading {CRYPTO} from {START_DATE} to {END_DATE}...")
    df = data_utils.load_crypto_data(CRYPTO, START_DATE, END_DATE)
    df = data_utils.add_features(df)
    if df.empty:
        print("❌ No data.")
        return None
    return df

def get_or_create_scalers(train_df):
    scaler_x, scaler_y = scaler_utils.load_scalers(prefix="crypto", in_dir=MODELS_DIR)
    if scaler_x is None or scaler_y is None:
        scaler_x = MinMaxScaler().fit(train_df[FEATURE_COLS])
        scaler_y = MinMaxScaler().fit(train_df[[TARGET_COL]])
        scaler_utils.save_scalers(scaler_x, scaler_y, prefix="crypto", out_dir=MODELS_DIR)
    return scaler_x, scaler_y

def scale_data(df, scaler_x, scaler_y):
    scaled_x = scaler_x.transform(df[FEATURE_COLS])
    scaled_y = scaler_y.transform(df[[TARGET_COL]])
    return scaled_x, scaled_y

def get_model(x_train):
    latest_version = model_utils.get_latest_version()
    version = latest_version + 1
    if latest_version > 0:
        prev_path = os.path.join(MODELS_DIR, f"v{latest_version}", "model.h5")
        print(f"🔄 Finetuning from v{latest_version} ({prev_path})")
        model = model_utils.load_existing_model(prev_path)
        model.compile(optimizer="adam", loss="mean_squared_error")
    else:
        print("🚀 Training new model...")
        model = model_utils.build_new_model((x_train.shape[1], x_train.shape[2]))
    return model, version

def train_and_save_model(model, x_train, y_train, version, scaler_x, scaler_y):
    callbacks = train_utils.get_callbacks(version)
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )
    model_utils.save_model_with_meta(
        model, scaler_x, scaler_y, version,
        history=history,
        config={"EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE, "PRE_DAY": PRE_DAY}
    )
    return model

def predict_and_evaluate(model, x_test, y_test, scaler_y, df, test_size):
    y_pred_test = model.predict(x_test)
    y_pred_test = scaler_y.inverse_transform(y_pred_test)
    y_true_test = scaler_y.inverse_transform(y_test)
    dates_test = df.index[-test_size:]
    metrics = analytic_utils.evaluate_predictions(y_true_test, y_pred_test)
    analytic_utils.print_evaluation(metrics, prefix="Test")
    return y_true_test, y_pred_test, dates_test

def predict_future(model, df, scaler_x, scaler_y):
    scaled_x = scaler_x.transform(df[FEATURE_COLS])
    last_window = scaled_x[-PRE_DAY:]
    preds_future = []
    current = last_window.copy()
    for _ in range(PREDICT_DAYS):
        pred = model.predict(current.reshape(1, PRE_DAY, len(FEATURE_COLS)))
        preds_future.append(pred[0, 0])
        new_row = current[-1].copy()
        new_row[-1] = pred
        current = np.vstack([current[1:], new_row])
    preds_future = scaler_y.inverse_transform(np.array(preds_future).reshape(-1, 1))
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=PREDICT_DAYS, freq="D")
    print("🔮 Future predictions:")
    for d, p in zip(future_dates, preds_future):
        print(f"{d.date()} | {p[0]:.2f} USD")
    return future_dates, preds_future

def main():
    df = load_and_prepare_data()
    if df is None:
        return

    test_size = max(60, int(len(df) * TEST_RATIO))
    train_df = df.iloc[:-test_size]
    scaler_x, scaler_y = get_or_create_scalers(train_df)
    scaled_x, scaled_y = scale_data(df, scaler_x, scaler_y)

    x_train, y_train, x_test, y_test, test_size = train_utils.split_train_test(
        df, scaled_x, scaled_y, PRE_DAY, test_ratio=TEST_RATIO
    )

    print(f"📊 Dataset size: {len(df)} rows")
    print(f"📊 Train: {x_train.shape}, Test: {x_test.shape}")

    model, version = get_model(x_train)
    model = train_and_save_model(model, x_train, y_train, version, scaler_x, scaler_y)

    future_dates, preds_future = predict_future(model, df, scaler_x, scaler_y)
    
    y_true_test, y_pred_test, dates_test = predict_and_evaluate(
        model, x_test, y_test, scaler_y, df, test_size
    )
    plot_utils.plot_forecast(
        df, y_true_test, y_pred_test, dates_test,
        future_dates, preds_future, version, out_dir=MODELS_DIR
    )

if __name__ == "__main__":
    main()