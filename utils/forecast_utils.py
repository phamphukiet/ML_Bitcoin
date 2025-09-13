import numpy as np

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

