import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import (
    CRYPTO, START_DATE, END_DATE, PRE_DAY, TEST_RATIO,
    EPOCHS, BATCH_SIZE, MODELS_DIR, PREDICT_DAYS
)
from utils import (
    data_utils, scaler_utils, model_utils,
    train_utils, plot_utils, analytic_utils,
    forecast_utils, log_utils
)

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
    # Fit scaler trên tập train để tránh leakage
    scaler_x, scaler_y = scaler_utils.get_fresh_scalers(
        train_df[FEATURE_COLS], train_df[[TARGET_COL]],
        prefix="crypto", out_dir=MODELS_DIR
    )
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

        RESET_MODEL = True  # 👈 config nhỏ, có thể đưa vào config.py
        if RESET_MODEL:
            print(f"🔄 Loading kiến trúc từ v{latest_version} và reset weight...")
            model = model_utils.load_existing_model_with_reset(prev_path)
        else:
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
    return model, history


def predict_and_evaluate(model, x_test, y_test, scaler_y, df, test_size):
    y_pred_test = model.predict(x_test)
    y_pred_test = scaler_y.inverse_transform(y_pred_test)
    y_true_test = scaler_y.inverse_transform(y_test)
    dates_test = df.index[-test_size:]

    metrics = analytic_utils.evaluate_predictions(y_true_test, y_pred_test)
    analytic_utils.print_evaluation(metrics, prefix="Test")

    return y_true_test, y_pred_test, dates_test, metrics


def predict_future(model, df, scaler_x, scaler_y):
    preds_future = forecast_utils.robust_future_predict(
        model, df, scaler_x, scaler_y,
        pre_day=PRE_DAY, predict_days=PREDICT_DAYS, use_true_ratio=0.3
    )
    future_dates = pd.date_range(
        df.index[-1] + pd.Timedelta(days=1),
        periods=PREDICT_DAYS,
        freq="D"
    )

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

    # 1. Train model
    model, history = train_and_save_model(model, x_train, y_train, version, scaler_x, scaler_y)

    # 2. Evaluate
    y_true_test, y_pred_test, dates_test, metrics = predict_and_evaluate(
        model, x_test, y_test, scaler_y, df, test_size
    )

    # 3. Save model + scaler + metrics
    model_utils.save_model_with_meta(
        model, scaler_x, scaler_y, version,
        history=history,
        config={"EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE, "PRE_DAY": PRE_DAY},
        metrics=metrics
    )
    log_utils.save_metrics_to_csv(metrics, version, MODELS_DIR)

    # 4. Forecast tương lai
    future_dates, preds_future = predict_future(model, df, scaler_x, scaler_y)
    forecast_utils.save_future_forecast(preds_future, future_dates, version, out_dir=MODELS_DIR)
    
    # 5. Plot
    plot_utils.plot_forecast(
        df, y_true_test, y_pred_test, dates_test,
        future_dates, preds_future, version, out_dir=MODELS_DIR
    )

if __name__ == "__main__":
    for _ in range(5):  # chạy thử ít hơn để kiểm chứng
        main()
