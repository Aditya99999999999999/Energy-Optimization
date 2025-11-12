"""
Construction Progress & Delay Prediction
- Synthetic dataset: BIM updates + sensor streams (daily timesteps)
- Targets: next-day percent_complete (regression) and delay_flag (binary)
- Models:
    * LSTM (baseline)
    * CNN + LSTM hybrid
- Evaluation: MSE/R2 for regression, AUC/accuracy/classification report for classification
Author: ChatGPT (GPT-5 Thinking mini)
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ML libs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D, TimeDistributed, BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping

# reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# 1) Synthetic data generator
# -------------------------
def generate_synthetic_sites(n_sites=600, days=90):
    """
    Generates synthetic daily time-series per site/project:
      - BIM updates: num_elements_completed_today (cumulative available)
      - Sensor streams: worker_count, equipment_runtime_hours, vibration_level, ambient_temp
    Produces:
      - X_timeseries: shape (n_sites, days, n_features)
      - meta/static features DF
      - targets:
         * y_next_pct: next-day percent_complete (regression)
         * y_delay_flag: whether project ultimately experienced >7 day delay (binary)
    """
    sites_meta = []
    X_all = []
    y_next_pct = []
    y_delay_flag = []

    for i in range(n_sites):
        # site/project static meta
        site_id = f"S{i:04d}"
        gross_floor_area = np.random.uniform(800, 30000)
        complexity = np.clip(np.random.beta(2,4)*12 + np.random.randint(1,8), 0, 20)
        planned_duration_days = int(np.clip(gross_floor_area / 200 * np.random.uniform(3,6), 30, 540))
        # risk factor influences delays and sensor noise
        risk = np.clip(np.random.normal(loc=0.15 * (complexity/10), scale=0.07), 0, 1.0)

        # simulate baseline progress curve (sigmoid-like)
        days_arr = np.arange(days)
        mid = planned_duration_days * 0.5
        planned_progress = 1 / (1 + np.exp(-0.12 * (days_arr - mid)))  # 0..1
        planned_progress = np.clip(planned_progress, 0, 1)

        # dynamic actual progress influenced by interruptions and design-changes
        actual_progress = np.copy(planned_progress)
        # inject random interruptions/shocks
        n_shocks = np.random.poisson(lam=1 + risk*3)
        for _ in range(n_shocks):
            pos = np.random.randint(int(days*0.1), days-1)
            severity = np.random.uniform(0.05, 0.35) * (1 + risk)
            decay = np.exp(-np.linspace(0, 3, days - pos))
            actual_progress[pos:] -= severity * decay
        # small noise and clip
        actual_progress += np.random.normal(0, 0.01, size=days)
        actual_progress = np.clip(actual_progress, 0, 1)

        # build sensors correlated with progress:
        # worker_count increases with progress (more trades) but dips when shocks occur
        base_workers = np.clip(5 + (gross_floor_area/2000) + complexity*0.5, 5, 200)
        worker_count = base_workers * (0.5 + actual_progress) + np.random.normal(0, 2, size=days)
        worker_count = np.clip(worker_count, 0, None)

        # equipment runtime in hours/day proportional to worker_count but with noise
        equipment_runtime = np.clip(worker_count * np.random.uniform(0.6, 1.2) + np.random.normal(0, 1, size=days), 0, 24)

        # vibration_level: proxy for heavy work activity (higher when many heavy activities)
        vibration = 0.1 + 2.0 * actual_progress * np.random.uniform(0.6, 1.4) + np.random.normal(0, 0.05, size=days)

        # ambient temp seasonal-ish
        temp = 20 + 6 * np.sin((days_arr / 365.0) * 2*np.pi) + np.random.normal(0, 1.5, size=days)

        # BIM updates: elements completed per day (delta of percent*estimated_total_elements)
        estimated_total_elements = int(np.clip(gross_floor_area / 2.5 + complexity*10, 100, 50000))
        cumulative_completed = (actual_progress * estimated_total_elements).astype(int)
        # make per-day delta (ensuring non-negative)
        delta_completed = np.diff(np.concatenate([[0], cumulative_completed])).astype(int)
        delta_completed = np.clip(delta_completed, 0, None)

        # assemble feature matrix: we will include percent_complete_so_far and raw sensors and BIM delta
        percent_complete = actual_progress  # 0..1
        features = np.stack([
            percent_complete,               # current percent complete
            delta_completed,                # elements completed today
            worker_count,
            equipment_runtime,
            vibration,
            temp
        ], axis=1)  # shape (days, 6)

        # Determine final labels (simulate a delay flag if persistent lag or large shocks)
        # If final percent never reaches 0.99 by planned_duration_days + slack => delayed
        slack_days = 7
        target_finish_index = min(days-1, planned_duration_days + slack_days)
        finished_by_planned = actual_progress[target_finish_index] >= 0.99
        delay_flag = 0 if finished_by_planned else 1
        # random extra chance proportional to risk
        if np.random.rand() < 0.02 + 0.5*risk:
            delay_flag = 1

        # next-day percent target (we will predict percent at day t+1 using days-1 window)
        # to build sample sequences, we'll produce many sliding windows later
        sites_meta.append({
            "site_id": site_id,
            "gross_floor_area": gross_floor_area,
            "complexity": complexity,
            "planned_duration_days": planned_duration_days,
            "estimated_total_elements": estimated_total_elements,
            "risk": risk,
            "delay_flag": delay_flag
        })
        X_all.append(features)  # per-site full days x features
        y_next_pct.append(percent_complete)  # store full array (we'll slice later)
        y_delay_flag.append(delay_flag)

    # convert to arrays
    X_all = np.array(X_all)  # (n_sites, days, n_feats)
    y_next_pct = np.array(y_next_pct)  # (n_sites, days)
    y_delay_flag = np.array(y_delay_flag)

    meta_df = pd.DataFrame(sites_meta)
    return X_all, y_next_pct, y_delay_flag, meta_df

# -------------------------
# 2) Create supervised sliding-window dataset
# -------------------------
def build_sliding_windows(X_sites, y_pct_sites, meta_df, window_len=14, predict_horizon=1):
    """
    For each site, build sliding windows of length window_len (days).
    Input X windows: last `window_len` days of features (multivariate)
    Targets: percent_complete at t+predict_horizon (regression), and site-level delay_flag (classification)
    Returns arrays: X (n_samples, window_len, n_feats), y_reg (n_samples,), y_clf (n_samples,), meta_features (n_samples, ...)
    """
    n_sites, days, n_feats = X_sites.shape
    X_list, y_reg_list, y_clf_list, meta_list = [], [], [], []
    for i in range(n_sites):
        for end_day in range(window_len-1, days - predict_horizon):
            start = end_day - (window_len - 1)
            Xw = X_sites[i, start:end_day+1, :].copy()
            # regression target is percent_complete at end_day+predict_horizon
            y_reg = y_pct_sites[i, end_day + predict_horizon]
            y_clf = meta_df.loc[i, "delay_flag"]
            X_list.append(Xw)
            y_reg_list.append(y_reg)
            y_clf_list.append(y_clf)
            # meta features attached to this sample: planned_duration_days, complexity, risk, area
            meta_list.append([
                meta_df.loc[i, "gross_floor_area"],
                meta_df.loc[i, "complexity"],
                meta_df.loc[i, "planned_duration_days"],
                meta_df.loc[i, "risk"],
                meta_df.loc[i, "estimated_total_elements"]
            ])
    X = np.array(X_list)
    y_reg = np.array(y_reg_list)
    y_clf = np.array(y_clf_list)
    meta = np.array(meta_list)
    return X, y_reg, y_clf, meta

# -------------------------
# 3) Build models
# -------------------------
def build_lstm_model(input_shape, meta_shape=None, regression=True):
    """
    Simple LSTM model that can also take meta/static features via concatenation.
    input_shape = (window_len, n_feats)
    meta_shape = (n_meta,) or None
    """
    seq_in = Input(shape=input_shape, name="seq_in")
    x = LSTM(64, return_sequences=True)(seq_in)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)

    if meta_shape is not None:
        meta_in = Input(shape=(meta_shape,), name="meta_in")
        concat = concatenate([x, meta_in])
        h = Dense(32, activation="relu")(concat)
        h = Dropout(0.2)(h)
        inputs = [seq_in, meta_in]
    else:
        h = Dense(32, activation="relu")(x)
        h = Dropout(0.2)(h)
        inputs = seq_in

    if regression:
        out = Dense(1, activation="linear", name="pct_out")(h)
        model = Model(inputs=inputs, outputs=out)
        model.compile(optimizer="adam", loss="mse")
    else:
        out = Dense(1, activation="sigmoid", name="delay_out")(h)
        model = Model(inputs=inputs, outputs=out)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])

    return model

def build_cnn_lstm_model(input_shape, meta_shape=None, regression=True):
    """
    CNN+LSTM hybrid: apply Conv1D over time to learn local temporal features, then LSTM.
    """
    seq_in = Input(shape=input_shape, name="seq_in")
    # Conv stack
    c = Conv1D(64, kernel_size=3, padding="same", activation="relu")(seq_in)
    c = BatchNormalization()(c)
    c = Conv1D(32, kernel_size=3, padding="same", activation="relu")(c)
    c = BatchNormalization()(c)
    # LSTM on conv features
    x = LSTM(48)(c)
    x = Dropout(0.2)(x)

    if meta_shape is not None:
        meta_in = Input(shape=(meta_shape,), name="meta_in")
        concat = concatenate([x, meta_in])
        h = Dense(32, activation="relu")(concat)
        h = Dropout(0.2)(h)
        inputs = [seq_in, meta_in]
    else:
        h = Dense(32, activation="relu")(x)
        h = Dropout(0.2)(h)
        inputs = seq_in

    if regression:
        out = Dense(1, activation="linear", name="pct_out")(h)
        model = Model(inputs=inputs, outputs=out)
        model.compile(optimizer="adam", loss="mse")
    else:
        out = Dense(1, activation="sigmoid", name="delay_out")(h)
        model = Model(inputs=inputs, outputs=out)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])

    return model

# -------------------------
# 4) Training & evaluation helpers
# -------------------------
def evaluate_regression(y_true, y_pred, label="reg"):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"[{label}] MSE={mse:.6f}, R2={r2:.4f}")

def evaluate_classification(y_true, y_prob, name="clf"):
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float("nan")
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    print(f"[{name}] AUC={auc:.4f}, Acc={acc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

# -------------------------
# 5) Main workflow
# -------------------------
def main():
    print("Generating synthetic data...")
    X_sites, y_pct_sites, y_delay_sites, meta_df = generate_synthetic_sites(n_sites=600, days=90)

    # sliding windows
    window_len = 14
    predict_horizon = 1  # next-day percent
    X, y_reg, y_clf, meta = build_sliding_windows(X_sites, y_pct_sites, meta_df, window_len=window_len, predict_horizon=predict_horizon)
    print("Dataset shapes:", X.shape, y_reg.shape, y_clf.shape, meta.shape)

    # train/val/test split (sample-level)
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y_clf)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=SEED, stratify=y_clf[train_idx])  # 0.125*0.8 ~ 0.1

    # scale features (fit on train)
    n_samples, wlen, n_feats = X.shape
    X_flat = X.reshape((n_samples*wlen, n_feats))
    feat_scaler = StandardScaler()
    X_flat_scaled = feat_scaler.fit_transform(X_flat)
    X_scaled = X_flat_scaled.reshape((n_samples, wlen, n_feats))

    # meta scaling
    meta_scaler = StandardScaler()
    meta_scaled = meta_scaler.fit_transform(meta)

    # split
    X_train, X_val, X_test = X_scaled[train_idx], X_scaled[val_idx], X_scaled[test_idx]
    m_train, m_val, m_test = meta_scaled[train_idx], meta_scaled[val_idx], meta_scaled[test_idx]
    y_reg_train, y_reg_val, y_reg_test = y_reg[train_idx], y_reg[val_idx], y_reg[test_idx]
    y_clf_train, y_clf_val, y_clf_test = y_clf[train_idx], y_clf[val_idx], y_clf[test_idx]

    print("Training LSTM regression (predict next-day percent_complete)...")
    lstm_reg = build_lstm_model(input_shape=(window_len, n_feats), meta_shape=m_train.shape[1], regression=True)
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    lstm_reg.fit([X_train, m_train], y_reg_train, validation_data=([X_val, m_val], y_reg_val),
                 epochs=60, batch_size=64, callbacks=[es], verbose=1)

    # predict regression
    pred_reg_lstm = lstm_reg.predict([X_test, m_test]).flatten()
    evaluate_regression(y_reg_test, pred_reg_lstm, label="LSTM_reg (pct)")

    print("Training CNN+LSTM regression...")
    cnnl_reg = build_cnn_lstm_model(input_shape=(window_len, n_feats), meta_shape=m_train.shape[1], regression=True)
    es2 = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    cnnl_reg.fit([X_train, m_train], y_reg_train, validation_data=([X_val, m_val], y_reg_val),
                 epochs=60, batch_size=64, callbacks=[es2], verbose=1)
    pred_reg_cnnl = cnnl_reg.predict([X_test, m_test]).flatten()
    evaluate_regression(y_reg_test, pred_reg_cnnl, label="CNNL_reg (pct)")

    # combine reg predictions (simple average)
    pred_reg_stack = 0.5 * pred_reg_lstm + 0.5 * pred_reg_cnnl
    evaluate_regression(y_reg_test, pred_reg_stack, label="Ensemble_reg (avg)")

    # -------------------------
    # classification (delay flag)
    # -------------------------
    print("Training LSTM classifier for delay_flag...")
    lstm_clf = build_lstm_model(input_shape=(window_len, n_feats), meta_shape=m_train.shape[1], regression=False)
    es3 = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    lstm_clf.fit([X_train, m_train], y_clf_train, validation_data=([X_val, m_val], y_clf_val),
                 epochs=60, batch_size=64, callbacks=[es3], verbose=1)

    pred_clf_lstm = lstm_clf.predict([X_test, m_test]).flatten()
    evaluate_classification(y_clf_test, pred_clf_lstm, name="LSTM_clf")

    print("Training CNN+LSTM classifier for delay_flag...")
    cnnl_clf = build_cnn_lstm_model(input_shape=(window_len, n_feats), meta_shape=m_train.shape[1], regression=False)
    es4 = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    cnnl_clf.fit([X_train, m_train], y_clf_train, validation_data=([X_val, m_val], y_clf_val),
                 epochs=60, batch_size=64, callbacks=[es4], verbose=1)
    pred_clf_cnnl = cnnl_clf.predict([X_test, m_test]).flatten()
    evaluate_classification(y_clf_test, pred_clf_cnnl, name="CNNL_clf")

    # ensemble: average probabilities
    pred_clf_stack = 0.5 * pred_clf_lstm + 0.5 * pred_clf_cnnl
    evaluate_classification(y_clf_test, pred_clf_stack, name="Ensemble_clf (avg)")

    # -------------------------
    # Quick visualizations
    # -------------------------
    # Regression scatter
    plt.figure(figsize=(6,4))
    plt.scatter(y_reg_test, pred_reg_stack, alpha=0.4, s=10)
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel("Actual next-day percent_complete")
    plt.ylabel("Predicted (ensemble)")
    plt.title("Regression: actual vs predicted (next-day percent)")
    plt.tight_layout()
    plt.show()

    # Classification: top flagged examples
    top_idx = np.argsort(pred_clf_stack)[-10:]
    print("\nTop 10 samples by predicted delay probability (test set indices -> actual flag):")
    for k in top_idx:
        print(f"idx={test_idx[k]}, prob={pred_clf_stack[k]:.3f}, actual={y_clf_test[k]}")

    # show time-series example for a single test sample
    sample = 3
    fig, ax = plt.subplots(2,1,figsize=(9,5), sharex=True)
    ax[0].plot(X_test[sample,:,0], label="percent_complete (input)")
    ax[0].plot(range(window_len, window_len+1), [y_reg_test[sample]], 'ro', label="actual next-day")
    ax[0].legend()
    ax[1].plot(X_test[sample,:,2], label="worker_count (input)")
    ax[1].legend()
    plt.suptitle(f"Example test sample (pred_pct={pred_reg_stack[sample]:.3f}, delay_prob={pred_clf_stack[sample]:.3f})")
    plt.tight_layout()
    plt.show()

    print("Done.")

if __name__ == "__main__":
    main()
