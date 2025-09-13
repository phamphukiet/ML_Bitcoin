import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_predictions(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    }

def print_evaluation(metrics, prefix="Test"):
    print(f"📊 {prefix} evaluation:")
    for k, v in metrics.items():
        if k in ["RMSE", "MAE"]:
            print(f"   {k}: {v:,.2f} USD")
        elif k == "MAPE":
            print(f"   {k}: {v:.2f}%")
        else:
            print(f"   {k}: {v:.4f}")
