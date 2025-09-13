import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def plot_forecast(df, y_true_test, y_pred_test, dates_test,
                future_dates, preds_future, version, out_dir="models"):

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Thực tế toàn bộ
    plt.plot(df.index, df["Close"], color="blue", label="Thực tế")

    # Dự đoán test
    plt.plot(dates_test, y_pred_test, color="orange", linestyle="--", label="Dự đoán (test)")

    # Dự đoán tương lai
    plt.plot(future_dates, preds_future, color="red", linestyle="--", marker="o", label="Dự đoán (tương lai)")

    # Vạch đánh dấu END_DATE
    plt.axvline(df.index[-1], color="gray", linestyle=":", label="END_DATE")

    # Format trục Y thành USD
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

    plt.title(f"Dự báo giá {df.columns.name or 'crypto'} - Model v{version}")
    plt.xlabel("Ngày")
    plt.ylabel("Giá (USD)")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(plots_dir, f"forecast_v{version}.png")
    plt.savefig(path)
    plt.close()
    print(f"📊 Đã lưu biểu đồ dự báo: {path}")
