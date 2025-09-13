import os
import csv

def save_metrics_to_csv(metrics, version, out_dir="models"):
    """Ghi metrics vào file CSV để tiện tra cứu."""
    path = os.path.join(out_dir, "metrics_log.csv")
    file_exists = os.path.isfile(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["version", "RMSE", "MAE", "MAPE", "R2"])
        writer.writerow([
            version,
            round(metrics["RMSE"], 2),
            round(metrics["MAE"], 2),
            round(metrics["MAPE"], 2),
            round(metrics["R2"], 4)
        ])
    print(f"📑 Đã ghi metrics vào {path}")
