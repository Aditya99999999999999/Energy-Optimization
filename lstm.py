"""
Smart Energy Optimization with LSTM (Deep Learning)
---------------------------------------------------
Monitors and optimizes energy use on construction sites.
Applies:
- LSTM neural network for power prediction (per device)
- IsolationForest for anomaly detection
- Greedy optimization for suggested shutdowns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# -------------------------
# 1) Synthetic data generator (same idea as before)
# -------------------------
def generate_synthetic_telemetry(n_days=30, freq_mins=60):
    devices = {
        "generator_main": ("generator", 80, 20, True),
        "lights_zone_A": ("lights", 10, 3, False),
        "lights_zone_B": ("lights", 8, 2.5, False),
        "excavator_1": ("heavy", 45, 15, True),
        "crane_1": ("heavy", 50, 18, True),
        "compressor_1": ("heavy", 25, 8, False),
    }

    start = datetime.now() - timedelta(days=n_days)
    timestamps = [start + timedelta(minutes=freq_mins * i) for i in range(int((n_days * 24 * 60) / freq_mins))]

    rows = []
    for ts in timestamps:
        hour = ts.hour
        site_temp = 25 + 7 * np.sin((ts.timetuple().tm_yday / 365) * 2 * np.pi) + np.random.randn() * 1.5
        humidity = 50 + 10 * np.cos((hour / 24) * 2 * np.pi) + np.random.randn() * 5

        for dev, (dtype, base, var, critical) in devices.items():
            if dtype == "heavy":
                status = np.random.rand() < (0.6 if 6 <= hour <= 18 else 0.15)
            elif dtype == "lights":
                status = np.random.rand() < (0.8 if (hour < 6 or hour > 18) else 0.2)
            else:
                status = np.random.rand() < 0.5

            power = 0
            if status:
                power = max(0.1, np.random.normal(base, var))
                if np.random.rand() < 0.005:
                    power *= np.random.uniform(1.5, 3.0)
                if dtype == "heavy" and site_temp > 30:
                    power *= 1.05

            rows.append({
                "timestamp": ts,
                "hour": hour,
                "dayofweek": ts.weekday(),
                "site_temp": site_temp,
                "humidity": humidity,
                "device_id": dev,
                "device_type": dtype,
                "power_kw": power,
                "status": int(status),
                "critical": int(critical)
            })

    df = pd.DataFrame(rows)
    agg = df.groupby("timestamp")["power_kw"].sum().reset_index().rename(columns={"power_kw": "site_total_kw"})
    df = df.merge(agg, on="timestamp")
    return df


# -------------------------
# 2) Data preparation for LSTM
# -------------------------
def prepare_lstm_sequences(df, seq_len=10):
    """
    Create LSTM-ready (X, y) sequences per device.
    seq_len = number of previous time steps per sequence.
    """
    device_dfs = []
    for device in df["device_id"].unique():
        sub = df[df["device_id"] == device].sort_values("timestamp")
        features = ["hour", "dayofweek", "site_temp", "humidity", "status", "critical", "site_total_kw"]
        X, y = [], []
        scaler = StandardScaler()
        sub_scaled = scaler.fit_transform(sub[features + ["power_kw"]])

        for i in range(len(sub_scaled) - seq_len):
            X.append(sub_scaled[i:i+seq_len, :-1])  # features
            y.append(sub_scaled[i+seq_len, -1])     # power target

        if len(X) > 0:
            X, y = np.array(X), np.array(y)
            device_dfs.append((device, X, y, scaler))

    return device_dfs


# -------------------------
# 3) LSTM model builder
# -------------------------
def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# -------------------------
# 4) Train & evaluate LSTM
# -------------------------
def train_lstm(device_data):
    models = {}
    for device, X, y, scaler in device_data:
        print(f"\nTraining LSTM for {device} ...")
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = build_lstm(input_shape=(X.shape[1], X.shape[2]))
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=25, batch_size=32, verbose=1, callbacks=[es])

        preds = model.predict(X_test).flatten()
        mse = np.mean((preds - y_test)**2)
        print(f"{device} → Test MSE: {mse:.5f}")
        models[device] = (model, scaler)
    return models


# -------------------------
# 5) Anomaly Detection
# -------------------------
def detect_anomalies(df):
    features = ["hour", "site_temp", "humidity", "power_kw"]
    X = df[features]
    iso = IsolationForest(n_estimators=150, contamination=0.02, random_state=42)
    df["anomaly_score"] = iso.decision_function(X)
    df["is_anomaly"] = iso.predict(X) == -1
    print(f"Detected anomalies: {df['is_anomaly'].sum()} rows (~{df['is_anomaly'].mean()*100:.2f}%)")
    return df


# -------------------------
# 6) Energy optimization
# -------------------------
def optimize_shutdowns(df, threshold_kw=120):
    actions = []
    for ts, group in df.groupby("timestamp"):
        total_kw = group["power_kw"].sum()
        if total_kw <= threshold_kw:
            continue
        noncritical = group[(group["critical"] == 0) & (group["status"] == 1)].sort_values("power_kw", ascending=False)
        saved_kw = 0
        for _, row in noncritical.iterrows():
            if total_kw - saved_kw <= threshold_kw:
                break
            saved_kw += row["power_kw"]
            actions.append({"timestamp": ts, "device_id": row["device_id"], "suggestion": "shutdown"})
    print(f"Proposed {len(actions)} shutdown actions.")
    return pd.DataFrame(actions)


# -------------------------
# 7) Main workflow
# -------------------------
def main():
    print("Generating synthetic data...")
    df = generate_synthetic_telemetry(n_days=14)

    print("Preparing sequences for LSTM...")
    device_data = prepare_lstm_sequences(df, seq_len=10)

    print("Training LSTM models...")
    models = train_lstm(device_data)

    print("Detecting anomalies...")
    df = detect_anomalies(df)

    print("Optimizing shutdowns...")
    actions = optimize_shutdowns(df)

    print("\nSample Optimization Actions:")
    print(actions.head(10))

    # Visualization: show one device pattern
    sample_dev = "lights_zone_A"
    sub = df[df["device_id"] == sample_dev]
    plt.figure(figsize=(8,3))
    plt.plot(sub["timestamp"], sub["power_kw"], label="Actual Power (kW)")
    plt.title(f"Energy Usage for {sample_dev}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
