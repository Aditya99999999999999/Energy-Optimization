"""
AI-Based BIM Model Validator
- Synthetic BIM generator (components, AABB bounding boxes, edit history)
- Deterministic clash detection (AABB)
- Feature engineering
- ML: RandomForest classifier on static features
- Deep Learning: LSTM on component edit-history to predict faults
- Anomaly detection: IsolationForest
- Ensemble/report of flagged components and suggested actions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# -------------------------
# 1) Synthetic BIM dataset generator
# -------------------------
def generate_synthetic_bim(n_components=500, max_edits=20, project_days=90):
    """
    Simulate BIM components:
    - component_id
    - type (wall, column, beam, duct, pipe, window, door, slab)
    - bbox: (xmin,ymin,zmin,xmax,ymax,zmax)
    - level (floor number)
    - material (categorical)
    - owner/trade
    - creation_time
    - edits: list of (timestamp, status, some numeric properties)
    - label: faulty (1) / ok (0) — we create faults by injecting collisions, wrong level, missing supports etc.
    """
    types = ["wall", "column", "beam", "duct", "pipe", "window", "door", "slab"]
    materials = ["concrete", "steel", "wood", "glass", "gyp", "insulation"]
    trades = ["structural", "MEP", "architectural"]

    components = []
    start_time = datetime.now() - timedelta(days=project_days)

    # helper to create bounding boxes within project extents
    def rand_box():
        # project extents 0..100 x/y, levels from 0..30 (z)
        x1 = random.uniform(0, 90)
        y1 = random.uniform(0, 90)
        z1 = random.uniform(0, 20)
        # size depends on type
        dx = random.uniform(0.5, 8.0)
        dy = random.uniform(0.5, 8.0)
        dz = random.uniform(0.2, 4.0)
        return (x1, y1, z1, x1 + dx, y1 + dy, z1 + dz)

    for i in range(n_components):
        ctype = random.choice(types)
        material = random.choice(materials)
        trade = random.choice(trades)
        level = random.randint(0, 10)
        bbox = rand_box()
        # jitter the Z to match level roughly
        zshift = level * 2.8  # floor-to-floor distance approximation
        bbox = (bbox[0], bbox[1], bbox[2] + zshift, bbox[3], bbox[4], bbox[5] + zshift)

        creation = start_time + timedelta(days=random.uniform(0, project_days))
        n_edits = random.randint(1, max_edits)
        edits = []
        for e in range(n_edits):
            ts = creation + timedelta(days=random.uniform(0, project_days - (creation - start_time).days))
            # status: 1 = present/on, 0 = removed, 2 = modified
            status = random.choices([1,2,0], weights=[0.8,0.15,0.05])[0]
            # some numeric properties that could drift (thickness, offset, rotation)
            thickness = random.uniform(0.1, 1.0) if ctype in ("wall","slab") else random.uniform(0.05, 0.5)
            offset = random.uniform(-0.3, 0.3)
            edits.append({"ts": ts, "status": status, "thickness": thickness, "offset": offset})

        components.append({
            "component_id": f"C{i:04d}",
            "type": ctype,
            "material": material,
            "trade": trade,
            "level": level,
            "bbox": bbox,  # tuple
            "creation": creation,
            "edits": sorted(edits, key=lambda x: x["ts"])
        })

    # Inject faults intentionally for labels
    #  - collisions (overlap with other components)
    #  - level mismatch (component at wrong level)
    #  - unsupported openings (window/door in slabs without support) [simulated]
    # We'll mark a subset as faulty and record reason
    n_faults = int(0.12 * n_components)
    faulty_indices = random.sample(range(n_components), n_faults)
    labels = np.zeros(n_components, dtype=int)
    fault_reasons = [None] * n_components

    # basic clash injection: take some components and move their bbox slightly to overlap with another
    for idx in faulty_indices[:int(0.5*n_faults)]:
        other = random.choice([j for j in range(n_components) if j != idx])
        ax = components[idx]
        bx = components[other]
        # move ax bbox center close to bx center
        b = bx["bbox"]
        ax_w = (ax["bbox"][3] - ax["bbox"][0])
        ax_h = (ax["bbox"][4] - ax["bbox"][1])
        ax_d = (ax["bbox"][5] - ax["bbox"][2])
        # new bbox centered near other's center
        cx = (b[0] + b[3]) / 2 + random.uniform(-0.5, 0.5)
        cy = (b[1] + b[4]) / 2 + random.uniform(-0.5, 0.5)
        cz = (b[2] + b[5]) / 2 + random.uniform(-0.5, 0.5)
        components[idx]["bbox"] = (cx-ax_w/2, cy-ax_h/2, cz-ax_d/2, cx+ax_w/2, cy+ax_h/2, cz+ax_d/2)
        labels[idx] = 1
        fault_reasons[idx] = "injected_clash"

    # level mismatch injection
    for idx in faulty_indices[int(0.5*n_faults):int(0.8*n_faults)]:
        components[idx]["level"] = components[idx]["level"] + random.choice([-2, -1, 1, 2])
        labels[idx] = 1
        fault_reasons[idx] = "level_mismatch"

    # unsupported opening injection for doors/windows
    for idx in faulty_indices[int(0.8*n_faults):]:
        if components[idx]["type"] in ("window", "door"):
            # mark as missing lintel/support (simulated)
            labels[idx] = 1
            fault_reasons[idx] = "unsupported_opening"

    # Attach labels/reasons to components
    for i, comp in enumerate(components):
        comp["faulty"] = int(labels[i])
        comp["fault_reason"] = fault_reasons[i]

    return components

# -------------------------
# 2) Geometric clash detection (AABB)
# -------------------------
def aabb_overlap(a, b, require_volume_overlap=False):
    """
    a and b are bbox tuples: (xmin,ymin,zmin,xmax,ymax,zmax)
    returns True if they overlap (touching counts)
    If require_volume_overlap=True, require positive intersection volume (not just touching)
    """
    axmin, aymin, azmin, axmax, aymax, azmax = a
    bxmin, bymin, bzmin, bxmax, bymax, bzmax = b

    x_overlap = (axmin < bxmax) and (axmax > bxmin)
    y_overlap = (aymin < bymax) and (aymax > bymin)
    z_overlap = (azmin < bzmax) and (azmax > bzmin)

    if require_volume_overlap:
        return x_overlap and y_overlap and z_overlap
    else:
        # allow touching -> use <=/>= accordingly
        return (axmin <= bxmax) and (axmax >= bxmin) and (aymin <= bymax) and (aymax >= bymin) and (azmin <= bzmax) and (azmax >= bzmin)

def detect_all_clashes(components):
    """
    Bruteforce O(N^2) clash detection — ok for hundreds to a few thousands of components.
    Returns list of (comp_id_a, comp_id_b) that clash.
    """
    clashes = []
    n = len(components)
    for i in range(n):
        a = components[i]["bbox"]
        for j in range(i+1, n):
            b = components[j]["bbox"]
            if aabb_overlap(a, b, require_volume_overlap=True):
                clashes.append((components[i]["component_id"], components[j]["component_id"]))
    return clashes

# -------------------------
# 3) Feature engineering
# -------------------------
def components_to_dataframe(components):
    rows = []
    for comp in components:
        xmin, ymin, zmin, xmax, ymax, zmax = comp["bbox"]
        vol = max(0, (xmax-xmin)*(ymax-ymin)*(zmax-zmin))
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        cz = (zmin + zmax) / 2
        age_days = (datetime.now() - comp["creation"]).days
        n_edits = len(comp["edits"])
        avg_thickness = np.mean([e["thickness"] for e in comp["edits"]]) if n_edits>0 else 0.0
        avg_offset = np.mean([e["offset"] for e in comp["edits"]]) if n_edits>0 else 0.0

        rows.append({
            "component_id": comp["component_id"],
            "type": comp["type"],
            "material": comp["material"],
            "trade": comp["trade"],
            "level": comp["level"],
            "vol": vol,
            "cx": cx, "cy": cy, "cz": cz,
            "age_days": age_days,
            "n_edits": n_edits,
            "avg_thickness": avg_thickness,
            "avg_offset": avg_offset,
            "faulty": comp["faulty"],
            "fault_reason": comp["fault_reason"]
        })
    df = pd.DataFrame(rows)
    # One-hot encode small categoricals
    df = pd.get_dummies(df, columns=["type","material","trade"], drop_first=True)
    return df

# -------------------------
# 4) Prepare sequences for LSTM (component edit history)
# -------------------------
def prepare_edit_sequences(components, seq_len=8, features_per_edit=["thickness","offset","status"]):
    """
    For each component, convert its edits into a fixed-length sequence (padded with zeros).
    We'll use recent seq_len edits (most recent last). If fewer edits, pad with zeros.
    Returns:
        X_seq: (n_components, seq_len, n_features)
        y: labels (faulty)
        ids: component ids (order matches rows)
    """
    X = []
    y = []
    ids = []
    for comp in components:
        edits = comp["edits"]
        seq = []
        for ed in edits[-seq_len:]:
            # features: thickness, offset, status (status numeric)
            seq.append([ed.get("thickness",0.0), ed.get("offset",0.0), ed.get("status",1)])
        # pad at front if shorter
        if len(seq) < seq_len:
            pad = [[0.0, 0.0, 0.0]] * (seq_len - len(seq))
            seq = pad + seq
        X.append(seq)
        y.append(comp["faulty"])
        ids.append(comp["component_id"])
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    return X, y, ids

# -------------------------
# 5) Build & train ML models
# -------------------------
def train_rf_classifier(df):
    X = df.drop(columns=["component_id","faulty","fault_reason"])
    y = df["faulty"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    print("RandomForest classification report:")
    print(classification_report(y_test, preds, digits=4))
    try:
        print("RF ROC AUC:", roc_auc_score(y_test, proba))
    except Exception:
        pass
    return model, X_train, X_test, y_train, y_test

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model

def train_lstm_model(X, y, epochs=30, batch_size=32):
    """
    X: (n, seq_len, n_features)
    y: (n,)
    """
    scaler = StandardScaler()
    # Flatten time dimension for scaling per-feature across dataset
    n, seq_len, nf = X.shape
    X_flat = X.reshape((n*seq_len, nf))
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape((n, seq_len, nf))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    model = build_lstm_model(input_shape=(seq_len, nf))
    es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=1)
    # evaluate
    eval_res = model.evaluate(X_test, y_test, verbose=0)
    print(f"LSTM eval -> loss: {eval_res[0]:.4f}, AUC: {eval_res[1]:.4f}")
    return model, scaler, (X_train, X_test, y_train, y_test, history)

# -------------------------
# 6) Anomaly detection (IsolationForest)
# -------------------------
def run_anomaly_detection(df):
    X = df.drop(columns=["component_id","faulty","fault_reason"])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=RANDOM_SEED)
    iso.fit(Xs)
    scores = iso.decision_function(Xs)
    preds = iso.predict(Xs)  # -1 anomaly, 1 normal
    df["anomaly_score"] = scores
    df["is_anomaly"] = (preds == -1).astype(int)
    print(f"IsolationForest flagged {df['is_anomaly'].sum()} anomalies (~{df['is_anomaly'].mean()*100:.2f}%).")
    return df, iso, scaler

# -------------------------
# 7) Combine results & report
# -------------------------
def assemble_report(components, df_features, rf_model, lstm_model_tuple, iso_df, X_seq_ids):
    # RF predictions
    X = df_features.drop(columns=["component_id","faulty","fault_reason"])
    rf_probs = rf_model.predict_proba(X)[:,1]
    df_features["rf_prob"] = rf_probs

    # LSTM predictions: need to align order of X_seq_ids with df rows
    X_seq, ids = X_seq_ids
    lstm_model, lstm_scaler = lstm_model_tuple
    n, seq_len, nf = X_seq.shape
    Xflat = X_seq.reshape((n*seq_len, nf))
    try:
        Xflat_scaled = lstm_scaler.transform(Xflat)
    except Exception:
        # if lstm_scaler None or different shape, fall back to identity
        Xflat_scaled = Xflat
    Xseq_scaled = Xflat_scaled.reshape((n, seq_len, nf))
    lstm_probs = lstm_model.predict(Xseq_scaled).flatten()

    # map lstm probs into df_features by component_id
    id_to_lstm = dict(zip(ids, lstm_probs))
    df_features["lstm_prob"] = df_features["component_id"].map(id_to_lstm).fillna(0.0)

    # unified score (simple)
    df_features["unified_score"] = 0.5 * df_features["rf_prob"] + 0.5 * df_features["lstm_prob"]
    # also include anomaly flag
    df_features["is_anomaly"] = iso_df["is_anomaly"].values

    # generate suggested actions:
    # - if unified_score > 0.6 or is_anomaly: flag for manual review and priority
    df_features["suggested_action"] = df_features.apply(
        lambda r: ("ESCALATE" if (r["unified_score"] > 0.7 or r["is_anomaly"]==1) else
                   ("REVIEW" if r["unified_score"] > 0.4 else "OK")), axis=1)

    # attach detected clashes (from deterministic function)
    clashes = detect_all_clashes(components)
    clash_dict = {}
    for a,b in clashes:
        clash_dict.setdefault(a, []).append(b)
        clash_dict.setdefault(b, []).append(a)

    df_features["detected_clashes"] = df_features["component_id"].map(lambda cid: clash_dict.get(cid, []))

    # Print top positives for inspection
    top = df_features.sort_values("unified_score", ascending=False).head(15)
    print("\nTop flagged components (by unified_score):")
    print(top[["component_id","rf_prob","lstm_prob","unified_score","is_anomaly","detected_clashes","suggested_action"]])

    return df_features, clashes

# -------------------------
# 8) Example / main
# -------------------------
def main():
    print("Generating synthetic BIM components...")
    comps = generate_synthetic_bim(n_components=800, max_edits=12, project_days=120)

    print("Detecting deterministic clashes (AABB)...")
    basic_clashes = detect_all_clashes(comps)
    print(f"Deterministic clashes found: {len(basic_clashes)} (pairwise)")

    print("Featurizing components...")
    df = components_to_dataframe(comps)

    print("Preparing edit history sequences for LSTM...")
    X_seq, y_seq, ids = prepare_edit_sequences(comps, seq_len=10)
    print(f"Prepared LSTM sequences shape: {X_seq.shape}, labels distribution: {np.bincount(y_seq)}")

    print("Training RandomForest classifier on static features...")
    rf_model, X_train, X_test, y_train, y_test = train_rf_classifier(df)

    print("Training LSTM on edit histories...")
    lstm_model, lstm_scaler, lstm_info = train_lstm_model(X_seq, y_seq, epochs=30, batch_size=64)

    print("Running anomaly detection (IsolationForest) on static features...")
    iso_df, iso_model, iso_scaler = run_anomaly_detection(df)

    print("Assembling report...")
    df_report, clashes = assemble_report(comps, df, rf_model, (lstm_model, lstm_scaler), iso_df, (X_seq, ids))

    # Quick visualization: histogram of unified score
    try:
        plt.figure(figsize=(6,3))
        plt.hist(df_report["unified_score"], bins=30)
        plt.title("Distribution of Unified Fault Scores")
        plt.xlabel("unified_score")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

    # Example: export flagged items to CSV
    flagged = df_report[(df_report["suggested_action"]!="OK")].sort_values("unified_score", ascending=False)
    print(f"\nFlagged components for action: {len(flagged)} (exporting sample 10 rows)")
    print(flagged[["component_id","unified_score","suggested_action","detected_clashes"]].head(10))

    # Save to disk
    flagged.to_csv("bim_flagged_components.csv", index=False)
    print("Saved flagged components to bim_flagged_components.csv")

if __name__ == "__main__":
    main()
