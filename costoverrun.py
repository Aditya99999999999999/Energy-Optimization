"""
Predict cost overruns and schedule delays using BIM parameters + project history
- Synthetic dataset generator
- Static-model: XGBoost (uses BIM-derived static features)
- Sequential-model: LSTM (uses weekly time-series of cost % and percent-complete)
- Stacking ensemble: simple logistic/regression on top of base preds
- Outputs evaluation metrics and example predictions
Author: ChatGPT (GPT-5 Thinking mini)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping

# reproducibility
RND = 42
np.random.seed(RND)
random.seed(RND)
tf.random.set_seed(RND)

# -------------------------
# 1) Synthetic data generator
# -------------------------
def generate_synthetic_projects(n_projects=800, weeks=40):
    """
    Creates synthetic projects with:
      - static BIM-like features (gross_floor_area, total_volume, element_counts, trades_count, complexity_score)
      - weekly time series: cumulative_cost, percent_complete
      - labels: cost_overrun_pct, cost_overrun_flag, schedule_delay_days, delay_flag
    Returns:
      projects_df, timeseries_array (n_projects, weeks, 2), ids
    """
    projects = []
    timeseries = []
    ids = []

    for i in range(n_projects):
        pid = f"P{i:04d}"
        ids.append(pid)

        # Static BIM-derived features
        gross_floor_area = np.random.uniform(500, 50000)  # m2
        total_volume = gross_floor_area * np.random.uniform(3, 5)  # m3
        n_elements = int(np.clip(np.random.normal(gross_floor_area/2, 300), 50, 30000))
        n_trades = random.randint(3, 12)
        complexity = np.clip(np.random.beta(2,5)*10 + (n_trades/3.0), 0, 15)  # 0-15
        design_changes_expected = np.random.poisson(2 + complexity/3.0)

        # baseline planned cost and duration derived from size/complexity
        base_cost = (gross_floor_area * np.random.uniform(800, 2200))  # currency units
        planned_duration_weeks = int(np.clip(gross_floor_area / 1000 * np.random.uniform(10, 25), 8, 104))

        # simulate risk factors
        risk_index = np.clip(np.random.normal(0.1 * complexity, 0.08), 0, 1.0)

        # Simulate weekly time-series: percent complete and cumulative cost
        # start from week 0 to weeks-1
        cumulative_cost = 0.0
        percent_complete = 0.0
        weekly_costs = []
        pct_series = []
        for w in range(weeks):
            # planned progress curve (sigmoid-ish)
            planned_pct = 1 / (1 + np.exp(-0.14*(w - planned_duration_weeks/2)))  # normalized
            # actual progress lags or leads influenced by risk and random noise
            noise = np.random.normal(0, 0.02)
            progress = planned_pct - (risk_index * 0.1) + noise
            progress = np.clip(progress, 0.0, 1.0)
            # weekly incremental cost ~ fraction of base_cost times delta of progress
            weekly_fraction = max(0.0, progress - percent_complete)
            weekly_cost = weekly_fraction * base_cost * (1 + np.random.normal(0, 0.03) + 0.1*design_changes_expected/5.0)
            cumulative_cost += weekly_cost
            percent_complete = progress
            weekly_costs.append(cumulative_cost)
            pct_series.append(percent_complete)

        weekly_costs = np.array(weekly_costs)
        pct_series = np.array(pct_series)

        # Introduce potential overruns/delays by injecting shocks for some projects
        # Determine ground truth overrun and delay
        overrun_chance = 0.12 + 0.6 * risk_index  # riskier projects more likely to overrun
        delay_chance = 0.10 + 0.5 * risk_index

        cost_overrun_pct = np.random.normal(0.02, 0.02)  # baseline small chance
        schedule_delay_days = 0

        if np.random.rand() < overrun_chance:
            # inject unexpected costs late in the project
            shock_weeks = random.randint(int(weeks*0.3), weeks-1)
            shock_amount = base_cost * np.random.uniform(0.05, 0.30) * (1 + complexity/20)
            weekly_costs[shock_weeks:] += shock_amount
            cost_overrun_pct += shock_amount / base_cost

        if np.random.rand() < delay_chance:
            # delay adds to schedule (in weeks) and reduces progress rate
            delay_weeks = int(np.random.uniform(1, max(2, planned_duration_weeks*0.15)))
            schedule_delay_days = delay_weeks * 7
            # reduce later progress
            pct_series = np.clip(pct_series - np.linspace(0, 0.15, weeks), 0, 1.0)

        # finalize labels: compare final cumulative cost to base_cost
        final_cost = weekly_costs[-1]
        cost_overrun_pct = max(0.0, (final_cost - base_cost) / base_cost)  # non-negative
        cost_overrun_flag = int(cost_overrun_pct > 0.10)  # >10% flagged
        delay_flag = int(schedule_delay_days > 7)

        projects.append({
            "project_id": pid,
            "gross_floor_area": gross_floor_area,
            "total_volume": total_volume,
            "n_elements": n_elements,
            "n_trades": n_trades,
            "complexity": complexity,
            "design_changes_expected": design_changes_expected,
            "base_cost": base_cost,
            "planned_duration_weeks": planned_duration_weeks,
            "risk_index": risk_index,
            "cost_overrun_pct": cost_overrun_pct,
            "cost_overrun_flag": cost_overrun_flag,
            "schedule_delay_days": schedule_delay_days,
            "delay_flag": delay_flag
        })

        # timeseries: stack cumulative_cost and percent_complete as features
        ts = np.stack([weekly_costs / (base_cost + 1e-9), pct_series], axis=1)  # normalize cost by base_cost
        timeseries.append(ts)

    projects_df = pd.DataFrame(projects)
    timeseries_arr = np.array(timeseries)  # shape (n_projects, weeks, 2)
    return projects_df, timeseries_arr, [p["project_id"] for p in projects]

# -------------------------
# 2) Prepare data for models
# -------------------------
def prepare_static_features(df):
    X = df[["gross_floor_area","total_volume","n_elements","n_trades","complexity",
            "design_changes_expected","base_cost","planned_duration_weeks","risk_index"]].copy()
    # simple engineered features
    X["elem_density"] = X["n_elements"] / (X["gross_floor_area"] + 1e-9)
    X["cost_per_m2"] = X["base_cost"] / (X["gross_floor_area"] + 1e-9)
    return X

def prepare_lstm_data(ts_array, seq_len=20):
    """
    Convert raw timeseries (n_projects, weeks, n_feats) into LSTM-ready sequences.
    We will use the last `seq_len` weeks to predict final cost_overrun_pct and schedule delay.
    """
    n, weeks, nf = ts_array.shape
    if weeks < seq_len:
        # pad at front with zeros
        pad = np.zeros((n, seq_len - weeks, nf))
        ts_array = np.concatenate([pad, ts_array], axis=1)
        weeks = seq_len
    X_seq = ts_array[:, -seq_len:, :]  # take most recent `seq_len` weeks
    return X_seq

# -------------------------
# 3) Build & train models
# -------------------------
def train_xgb_models(X_train, y_reg_train, y_clf_train, X_val=None, y_reg_val=None, y_clf_val=None):
    params_reg = {"objective":"reg:squarederror", "n_estimators":200, "learning_rate":0.05, "random_state":RND}
    xgb_reg = xgb.XGBRegressor(**params_reg)
    xgb_reg.fit(X_train, y_reg_train, eval_set=[(X_val, y_reg_val)] if X_val is not None else None, verbose=False, early_stopping_rounds=25)

    params_clf = {"objective":"binary:logistic", "n_estimators":200, "learning_rate":0.05, "random_state":RND}
    xgb_clf = xgb.XGBClassifier(**params_clf)
    xgb_clf.fit(X_train, y_clf_train, eval_set=[(X_val, y_clf_val)] if X_val is not None else None, verbose=False, early_stopping_rounds=25)

    return xgb_reg, xgb_clf

def build_lstm_model(input_shape, regression=True):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    if regression:
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer="adam", loss="mse")
    else:
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model

# -------------------------
# 4) Ensemble stacking
# -------------------------
def stack_predictions(static_preds, seq_preds, y_true_reg=None, y_true_clf=None, task="regression"):
    """
    Simple stacking: linear model on top of base preds.
    static_preds, seq_preds: arrays shape (n_samples,)
    Returns stacked_pred (n_samples,) and trained meta model
    """
    X_stack = np.vstack([static_preds, seq_preds]).T
    if task == "regression":
        meta = LinearRegression()
        meta.fit(X_stack, y_true_reg)
        stacked = meta.predict(X_stack)
    else:
        meta = LogisticRegression(max_iter=200)
        meta.fit(X_stack, y_true_clf)
        stacked = meta.predict_proba(X_stack)[:,1]
    return stacked, meta

# -------------------------
# 5) Evaluation helpers
# -------------------------
def evaluate_regression(y_true, y_pred, name="regression"):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"[{name}] MSE: {mse:.6f}, R2: {r2:.4f}")

def evaluate_classification(y_true, y_prob, threshold=0.5, name="classification"):
    y_pred = (y_prob >= threshold).astype(int)
    auc = None
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = np.nan
    acc = accuracy_score(y_true, y_pred)
    print(f"[{name}] AUC: {auc:.4f}, Acc@{threshold}: {acc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

# -------------------------
# 6) Main workflow
# -------------------------
def main():
    print("Generating synthetic dataset...")
    projects_df, ts_array, ids = generate_synthetic_projects(n_projects=800, weeks=40)

    # Prepare features
    X_static = prepare_static_features(projects_df)
    X_seq = prepare_lstm_data(ts_array, seq_len=20)

    # Targets
    y_cost_overrun_pct = projects_df["cost_overrun_pct"].values  # regression
    y_cost_overrun_flag = projects_df["cost_overrun_flag"].values  # classification
    y_delay_days = projects_df["schedule_delay_days"].values
    y_delay_flag = projects_df["delay_flag"].values

    # Train/val/test split
    idx = np.arange(len(projects_df))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=RND, stratify=y_cost_overrun_flag)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=RND, stratify=y_cost_overrun_flag[train_idx])  # 0.125*0.8 ~ 0.1

    # Static feature scaling
    scaler = StandardScaler()
    X_static_scaled = scaler.fit_transform(X_static)

    Xs_train = X_static_scaled[train_idx]
    Xs_val = X_static_scaled[val_idx]
    Xs_test = X_static_scaled[test_idx]

    # LSTM input scaling (scale per feature across all time steps)
    n, seq_len, nf = X_seq.shape
    Xseq_flat = X_seq.reshape((n*seq_len, nf))
    seq_scaler = StandardScaler()
    Xseq_scaled_flat = seq_scaler.fit_transform(Xseq_flat)
    X_seq_scaled = Xseq_scaled_flat.reshape((n, seq_len, nf))

    Xl_train = X_seq_scaled[train_idx]
    Xl_val = X_seq_scaled[val_idx]
    Xl_test = X_seq_scaled[test_idx]

    # Targets per split
    y_reg_train = y_cost_overrun_pct[train_idx]
    y_reg_val = y_cost_overrun_pct[val_idx]
    y_reg_test = y_cost_overrun_pct[test_idx]

    y_clf_train = y_cost_overrun_flag[train_idx]
    y_clf_val = y_cost_overrun_flag[val_idx]
    y_clf_test = y_cost_overrun_flag[test_idx]

    # -------------------------
    # Train XGBoost models (static)
    # -------------------------
    print("Training XGBoost models on static BIM features...")
    xgb_reg, xgb_clf = train_xgb_models(Xs_train, y_reg_train, y_clf_train, Xs_val, y_reg_val, y_clf_val)

    static_reg_pred_test = xgb_reg.predict(Xs_test)
    static_clf_prob_test = xgb_clf.predict_proba(Xs_test)[:,1]

    evaluate_regression(y_reg_test, static_reg_pred_test, name="XGB_regressor (static)")
    evaluate_classification(y_clf_test, static_clf_prob_test, name="XGB_classifier (static)")

    # -------------------------
    # Train LSTM models (sequential)
    # -------------------------
    print("Training LSTM regression model on sequences (cost overrun pct)...")
    lstm_reg = build_lstm_model(input_shape=(seq_len, nf), regression=True)
    es = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
    lstm_reg.fit(Xl_train, y_reg_train, validation_data=(Xl_val, y_reg_val), epochs=60, batch_size=32, callbacks=[es], verbose=1)

    seq_reg_pred_test = lstm_reg.predict(Xl_test).flatten()
    evaluate_regression(y_reg_test, seq_reg_pred_test, name="LSTM_regressor (seq)")

    # Classification LSTM for overrun flag
    print("Training LSTM classifier on sequences (overrun flag)...")
    lstm_clf = build_lstm_model(input_shape=(seq_len, nf), regression=False)
    es2 = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
    lstm_clf.fit(Xl_train, y_clf_train, validation_data=(Xl_val, y_clf_val), epochs=60, batch_size=32, callbacks=[es2], verbose=1)

    seq_clf_prob_test = lstm_clf.predict(Xl_test).flatten()
    evaluate_classification(y_clf_test, seq_clf_prob_test, name="LSTM_classifier (seq)")

    # -------------------------
    # Stacking / ensemble for regression (cost_overrun_pct)
    # -------------------------
    print("Stacking predictions for regression (simple linear stack)...")
    stacked_reg, meta_reg = stack_predictions(static_reg_pred_test, seq_reg_pred_test, y_true_reg=y_reg_test, task="regression")
    evaluate_regression(y_reg_test, stacked_reg, name="Stacked_regression (static+seq)")

    # -------------------------
    # Stacking / ensemble for classification (overrun flag)
    # -------------------------
    print("Stacking predictions for classification (simple logistic stack)...")
    stacked_clf_prob, meta_clf = stack_predictions(static_clf_prob_test, seq_clf_pr_
