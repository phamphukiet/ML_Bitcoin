import os, random, numpy as np, tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config import MODELS_DIR

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def get_callbacks(version):
    """Trả về danh sách callbacks cho training, checkpoint lưu vào thư mục version."""
    version_dir = os.path.join(MODELS_DIR, f"v{version}")
    os.makedirs(version_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(version_dir, "best_model_tmp.h5"),
            monitor="val_loss",
            save_best_only=True
        )
    ]
    return callbacks

def split_train_test(df, scaled_x, scaled_y, pre_day, test_ratio=0.2, min_test_days=60):
    n = len(df)
    test_size = max(min_test_days, int(n * test_ratio))
    if test_size >= n - pre_day:
        test_size = max(1, n - pre_day - 1)

    x, y = [], []
    for i in range(pre_day, len(scaled_x)):
        x.append(scaled_x[i-pre_day:i])
        y.append(scaled_y[i])
    x, y = np.array(x), np.array(y)

    x_train, y_train = x[:-test_size], y[:-test_size]
    x_test, y_test = x[-test_size:], y[-test_size:]

    return x_train, y_train, x_test, y_test, test_size