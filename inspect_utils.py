# make_train_result_3_3_5.py
import matplotlib.pyplot as plt
import json
import os

HISTORY_PATH = "models/v5/history.json"
OUT_FIG = "figure_3_3_5.png"

def plot_training_result(history_path=HISTORY_PATH, out_path=OUT_FIG):
    if not os.path.exists(history_path):
        print("❌ Không tìm thấy history.json, hãy train model trước.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    epochs = range(1, len(loss)+1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, loss, "b-", label="Training Loss (MSE)")
    if val_loss:
        plt.plot(epochs, val_loss, "r--", label="Validation Loss (MSE)")

    plt.title("Kết quả huấn luyện mô hình (Loss vs Epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📊 Đã tạo Hình 3.3.5: {out_path}")

if __name__ == "__main__":
    plot_training_result()
