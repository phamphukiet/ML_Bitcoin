import os, json

def get_best_model(registry_path="models/registry.json", criterion="R2", mode="max"):
    if not os.path.exists(registry_path):
        raise FileNotFoundError("❌ registry.json không tồn tại")

    with open(registry_path, "r") as f:
        registry = json.load(f)

    best = None
    for entry in registry:
        metrics = entry.get("metrics", {})
        if criterion not in metrics:
            continue
        value = metrics[criterion]
        if best is None:
            best = entry
        else:
            if mode == "max" and value > best["metrics"][criterion]:
                best = entry
            elif mode == "min" and value < best["metrics"][criterion]:
                best = entry
    return best
