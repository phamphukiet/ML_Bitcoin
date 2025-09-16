import os, json, time, joblib
from tensorflow.keras.models import Sequential, load_model,clone_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from config import UNITS, MODELS_DIR

def build_new_model(input_shape):
    model = Sequential([
        LSTM(UNITS, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(UNITS, return_sequences=True),
        Dropout(0.2),
        LSTM(UNITS),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def get_latest_version():
    if not os.path.exists(MODELS_DIR):
        return 0
    versions = [int(d[1:]) for d in os.listdir(MODELS_DIR) if d.startswith("v") and d[1:].isdigit()]
    return max(versions) if versions else 0

def load_existing_model(path):
    return load_model(path, compile=False)

def save_model_with_meta(model, scaler_x, scaler_y, version, history=None, config=None, metrics=None):
    version_dir = os.path.join(MODELS_DIR, f"v{version}")
    os.makedirs(version_dir, exist_ok=True)

    # --- Save model
    model_path = os.path.join(version_dir, "model.keras")
    model.save(model_path, save_format="keras")


    # --- Save scalers
    joblib.dump(scaler_x, os.path.join(version_dir, "scaler_x.pkl"))
    joblib.dump(scaler_y, os.path.join(version_dir, "scaler_y.pkl"))

    # --- Save training history
    if history is not None:
        with open(os.path.join(version_dir, "history.json"), "w") as f:
            json.dump(history.history, f, indent=2)

    # --- Save metadata
    meta = {
        "version": version,
        "model_path": model_path,
        "scaler_x": os.path.join(version_dir, "scaler_x.pkl"),
        "scaler_y": os.path.join(version_dir, "scaler_y.pkl"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config or {},
        "metrics": metrics or {}   # 👈 thêm vào đây
    }
    with open(os.path.join(version_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- Update registry.json
    registry_path = os.path.join(MODELS_DIR, "registry.json")
    registry = []
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    registry.append(meta)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    # --- Update latest.json
    latest_path = os.path.join(MODELS_DIR, "latest.json")
    with open(latest_path, "w") as f:
        json.dump(meta, f, indent=2)

    return model_path

def load_existing_model_with_reset(path, input_shape=None):
    old_model = load_model(path, compile=False) 
    model = clone_model(old_model)   
    if input_shape is not None:
        try:
            model.build((None, input_shape[0], input_shape[1]))
        except Exception:
            pass 

    model.compile(optimizer="adam", loss="mean_squared_error")
    print("🔄 Loaded architecture từ model cũ, reset weights.")
    return model

