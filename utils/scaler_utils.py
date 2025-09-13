import joblib
from sklearn.preprocessing import MinMaxScaler

def save_scalers(scaler_x, scaler_y, prefix="scaler", out_dir="."):
    joblib.dump(scaler_x, f"{out_dir}/{prefix}_x.pkl")
    joblib.dump(scaler_y, f"{out_dir}/{prefix}_y.pkl")
    print(f"💾 Đã lưu scaler vào {out_dir}")

def load_scalers(prefix="scaler", in_dir="."):
    try:
        scaler_x = joblib.load(f"{in_dir}/{prefix}_x.pkl")
        scaler_y = joblib.load(f"{in_dir}/{prefix}_y.pkl")
        print("📂 Đã load scaler từ file.")
        return scaler_x, scaler_y
    except FileNotFoundError:
        print("⚠️ Không tìm thấy scaler cũ, sẽ tạo mới.")
        return None, None
