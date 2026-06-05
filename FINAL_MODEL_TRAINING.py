"""
FMCW RADAR — FINAL TRAINING SCRIPT (5 classes)
=================================================
Activities: sitting | standing | walking | running | falling

New features added (v2):
  1. Temporal rolling features  — var-of-std, range-of-mean, CV-of-std per window
  2. Savitzky-Golay noise filter — smooths raw signal before any extraction
  3. Low-frequency FFT bins     — bins 1-8 (≈0.1-2 Hz) capturing sway/breath periodicity

How to use
----------
1. Set BASE_PATH  → folder containing all your CSV trial files
2. Set OUTPUT_PATH → where the 6 .pkl files will be saved
3. Run the script.
4. Use test_single_final.py or realtime_final.py with the saved .pkl files.
"""

import os, glob, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter          # ← NEW: Savitzky-Golay
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import joblib, warnings
warnings.filterwarnings("ignore")

# ================================================================
# ⚙️  CONFIGURATION — only edit these
# ================================================================
BASE_PATH    = r"D:\FYP\dataset_50samples"
OUTPUT_PATH  = r"D:\FYP"
WINDOW_SIZE  = 50      # MUST match realtime_final.py and test_single_final.py
STEP_SIZE    = 10
CORR_THRESH  = 0.90
KURT_CLIP    = 20.0
MIN_SAMPLES  = 200     # augment any class with fewer windows than this
N_ITER       = 15      # 15 × 3 = 45 total fits (fast and reliable)
CV_FOLDS     = 3

# --- Savitzky-Golay filter parameters ----------------------------
# window_length must be odd and < WINDOW_SIZE; polyorder < window_length
SG_WINDOW    = 7       # smoothing window in samples (must be odd)
SG_POLYORDER = 2       # polynomial order for the filter

# --- Low-frequency FFT bins to extract (1-indexed, DC = bin 0) ---
# With WINDOW_SIZE=50 and 20 Hz sampling: bin k → k*0.4 Hz
# Bins 1-8 → 0.4 Hz – 3.2 Hz (covers breathing ~0.3 Hz, sway ~0.5-2 Hz)
LF_BINS      = list(range(1, 9))   # [1, 2, 3, 4, 5, 6, 7, 8]
# ================================================================

EXPECTED_FEATURES = [
    'mean_magnitude', 'std_magnitude', 'max_magnitude',
    'mean_phase', 'std_phase', 'energy', 'range_peak',
    'range_mean', 'range_std', 'doppler_mean', 'doppler_std',
    'peak_power', 'median_magnitude', 'percentile_25', 'percentile_75',
    'signal_entropy', 'zero_crossings', 'rms_value',
    'peak_to_avg_ratio', 'kurtosis_value'
]

LABEL_MAP = {
    'sitting_new' : 'sitting',
    'standing_new': 'standing',
    'sitting'     : 'sitting',
    'standing'    : 'standing',
    'walking'     : 'walking',
    'running'     : 'running',
    'falling'     : 'falling',
}

def ts(msg):
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}")

def section(n, title):
    print(f"\n{'='*62}\n  STAGE {n}/6 — {title}\n{'='*62}")

# ================================================================
# STAGE 1 — DISCOVER FILES
# ================================================================
section(1, "Discover CSV files")
t_start = time.time()
all_files = glob.glob(os.path.join(BASE_PATH, "**", "*.csv"), recursive=True)
ts(f"Found {len(all_files)} CSV files")
if not all_files:
    raise FileNotFoundError(f"No CSV files found in: {BASE_PATH}")

# ================================================================
# STAGE 2 — SLIDING-WINDOW FEATURE EXTRACTION
# ================================================================
section(2, "Sliding-window feature extraction")
ts(f"Window={WINDOW_SIZE} rows  Step={STEP_SIZE} rows  "
   f"SG filter: window={SG_WINDOW} poly={SG_POLYORDER}  "
   f"LF FFT bins: {LF_BINS}")

def extract_window(window_df):
    """
    Per-column statistics + 3 original engineered features
    + NEW: temporal rolling features, SG-smoothed stats,
           and low-frequency FFT bin powers.

    Keep this function IDENTICAL in realtime_final.py and
    test_single_final.py — any change here must be mirrored there.
    """
    agg = {}

    for col in window_df.columns:
        sig = pd.to_numeric(window_df[col].values, errors='coerce')
        sig = sig[~np.isnan(sig)]
        if len(sig) < 2:
            sig = np.zeros(WINDOW_SIZE)

        # ----------------------------------------------------------
        # A) Apply Savitzky-Golay filter to get a smoothed signal.
        #    Keeps peaks intact; removes high-freq noise that caused
        #    kurtosis spikes of 591 in walking windows.
        # ----------------------------------------------------------
        sg_win = min(SG_WINDOW, len(sig) if len(sig) % 2 == 1 else len(sig) - 1)
        sg_win = max(sg_win, SG_POLYORDER + 2 if (SG_POLYORDER + 2) % 2 == 1
                                                else SG_POLYORDER + 3)
        try:
            sig_smooth = savgol_filter(sig, window_length=sg_win,
                                       polyorder=SG_POLYORDER)
        except ValueError:
            sig_smooth = sig.copy()

        # ----------------------------------------------------------
        # B) Standard statistics (computed on the SMOOTHED signal)
        # ----------------------------------------------------------
        col_mean   = np.mean(sig_smooth)
        col_std    = np.std(sig_smooth)
        col_median = np.median(sig_smooth)
        col_max    = np.max(sig_smooth)
        col_min    = np.min(sig_smooth)
        col_iqr    = np.percentile(sig_smooth, 75) - np.percentile(sig_smooth, 25)
        col_energy = np.sum(sig_smooth ** 2)
        col_rms    = np.sqrt(np.mean(sig_smooth ** 2))
        col_kurt   = min(pd.Series(sig_smooth).kurt(), KURT_CLIP)

        agg[col + "_mean"]   = col_mean
        agg[col + "_std"]    = col_std
        agg[col + "_var"]    = np.var(sig_smooth)
        agg[col + "_median"] = col_median
        agg[col + "_max"]    = col_max
        agg[col + "_min"]    = col_min
        agg[col + "_range"]  = col_max - col_min
        agg[col + "_skew"]   = pd.Series(sig_smooth).skew()
        agg[col + "_kurt"]   = col_kurt
        agg[col + "_iqr"]    = col_iqr
        agg[col + "_energy"] = col_energy
        agg[col + "_rms"]    = col_rms

        # ----------------------------------------------------------
        # C) Full FFT statistics
        # ----------------------------------------------------------
        fft_v = np.abs(np.fft.fft(sig_smooth))
        agg[col + "_fft_mean"] = np.mean(fft_v)
        agg[col + "_fft_std"]  = np.std(fft_v)
        agg[col + "_fft_max"]  = np.max(fft_v)

        # ----------------------------------------------------------
        # D) ORIGINAL engineered features
        #    (kept for backward-compatibility with earlier pkl files)
        # ----------------------------------------------------------
        # 1. Excess kurtosis
        agg[col + "_kurt_excess"]   = col_kurt - 3.0
        # 2. IQR × energy
        agg[col + "_iqr_x_energy"]  = col_iqr * col_energy
        # 3. Stability (std / mean)
        agg[col + "_stability"]     = col_std / (abs(col_mean) + 1e-10)

        # ----------------------------------------------------------
        # E) NEW: Low-frequency FFT bin powers (bins 1–8)
        #    Captures breathing (~0.3 Hz) and body-sway (~0.5-2 Hz).
        #    Sitting vs standing differ markedly in these bands.
        # ----------------------------------------------------------
        # fft_v already computed above; bins are 1-indexed here
        for b in LF_BINS:
            if b < len(fft_v):
                agg[f"{col}_lf_fft_bin{b}"] = fft_v[b]
            else:
                agg[f"{col}_lf_fft_bin{b}"] = 0.0

        # ----------------------------------------------------------
        # F) NEW: Temporal / rolling features within the window
        #
        #    Divide the window into 5 equal sub-segments and compute:
        #      • variance  of sub-segment stds  → "how jittery is the motion?"
        #      • range     of sub-segment means  → "how much did the mean drift?"
        #      • CV        of sub-segment stds   → normalised jitter
        #
        #    Sitting/standing have low var-of-std (quiet signal).
        #    Walking/running have high, periodic var-of-std.
        #    Falling shows a sudden spike that gives a huge range-of-mean.
        # ----------------------------------------------------------
        n_segs = 5
        seg_len = max(1, len(sig_smooth) // n_segs)
        seg_stds  = []
        seg_means = []
        for s_idx in range(n_segs):
            seg = sig_smooth[s_idx * seg_len : (s_idx + 1) * seg_len]
            if len(seg) > 0:
                seg_stds.append(np.std(seg))
                seg_means.append(np.mean(seg))

        seg_stds  = np.array(seg_stds)  if seg_stds  else np.array([0.0])
        seg_means = np.array(seg_means) if seg_means else np.array([0.0])

        var_of_std   = np.var(seg_stds)
        range_of_mean = np.ptp(seg_means)            # max - min
        cv_of_std    = (np.std(seg_stds) /
                        (np.mean(seg_stds) + 1e-10))  # coefficient of variation

        agg[col + "_var_of_std"]    = var_of_std
        agg[col + "_range_of_mean"] = range_of_mean
        agg[col + "_cv_of_std"]     = cv_of_std

    return agg


def process_file(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "activity" not in df.columns:
        return []
    raw   = df["activity"].iloc[0].strip().lower()
    label = LABEL_MAP.get(raw, raw)
    avail = [c for c in EXPECTED_FEATURES if c in df.columns]
    if len(avail) < 10:
        return []
    df_f = df[avail].copy()
    rows = []
    for s in range(0, len(df_f) - WINDOW_SIZE + 1, STEP_SIZE):
        feat = extract_window(df_f.iloc[s:s + WINDOW_SIZE])
        feat["activity"] = label
        rows.append(feat)
    return rows


t_feat   = time.time()
all_rows = []
for i, f in enumerate(all_files):
    all_rows.extend(process_file(f))
    if (i + 1) % 10 == 0 or (i + 1) == len(all_files):
        elapsed = time.time() - t_feat
        eta = (len(all_files) - (i + 1)) / ((i + 1) / elapsed) if elapsed else 0
        ts(f"File {i+1}/{len(all_files)} | {len(all_rows)} windows | ETA {eta:.0f}s")

data = pd.DataFrame(all_rows)
ts(f"✅ {len(data)} windows × {data.shape[1]-1} features")
print("\n  Class distribution:")
for cls, cnt in data["activity"].value_counts().items():
    print(f"    {cls:<12}: {cnt:>5} windows")

# ================================================================
# STAGE 3 — AUGMENT SMALL CLASSES
# ================================================================
section(3, "Augment under-represented classes")
feat_cols = [c for c in data.columns if c != "activity"]
np.random.seed(42)

for cls in sorted(data["activity"].unique()):
    cnt = (data["activity"] == cls).sum()
    if cnt < MIN_SAMPLES:
        needed = MIN_SAMPLES - cnt
        sub    = data[data["activity"] == cls]
        aug    = []
        for _ in range(needed):
            row = sub.sample(1).copy()
            for col in feat_cols:
                row[col] += np.random.normal(0, abs(sub[col].std()) * 0.03)
            aug.append(row)
        data = pd.concat([data] + aug, ignore_index=True)
        ts(f"  '{cls}': {cnt} → {MIN_SAMPLES} (+{needed} augmented)")
    else:
        ts(f"  '{cls}': {cnt} (ok, no augmentation needed)")

# ================================================================
# STAGE 4 — CORRELATION FILTER + ENCODE + SCALE
# ================================================================
section(4, "Feature filtering and scaling")
X = data.drop(columns=["activity"])
y = data["activity"]

le    = LabelEncoder()
y_enc = le.fit_transform(y)
ts(f"Classes: {list(le.classes_)}")

ts("Computing correlation matrix...")
corr_mat = X.corr().abs()
upper    = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
to_drop  = [c for c in upper.columns if any(upper[c] > CORR_THRESH)]
X        = X.drop(columns=to_drop)
ts(f"Dropped {len(to_drop)} correlated → {X.shape[1]} features remaining")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
ts(f"Train: {len(X_train)} | Test: {len(X_test)}")

scaler  = RobustScaler()
X_tr_s  = scaler.fit_transform(X_train)
X_te_s  = scaler.transform(X_test)
ts("✅ RobustScaler fitted")

# ================================================================
# STAGE 5 — XGBOOST HYPERPARAMETER SEARCH
# ================================================================
section(5, "XGBoost hyperparameter search")
ts(f"n_iter={N_ITER}  cv={CV_FOLDS}  → {N_ITER*CV_FOLDS} total fits")
ts("Timing one fit to give you an ETA...")

probe = XGBClassifier(
    n_estimators=150, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="multi:softprob", num_class=len(np.unique(y_enc)),
    eval_metric="mlogloss", tree_method="hist",
    random_state=42, reg_alpha=0.5, reg_lambda=2.0)
t0  = time.time()
probe.fit(X_tr_s, y_train)
one = time.time() - t0
est = one * N_ITER * CV_FOLDS
ts(f"Single fit: {one:.2f}s → estimated total: {est:.0f}s (~{est/60:.1f} min)")
ts(f"With n_jobs=-1 (parallel): ~{est/60/4:.1f}–{est/60/2:.1f} min")
ts(f"ETA: {time.strftime('%H:%M:%S', time.localtime(time.time()+est/2))}")

print("\n  Each line below = 1 completed fit ↓\n")
sw = compute_sample_weight("balanced", y_train)

search = RandomizedSearchCV(
    estimator=XGBClassifier(
        objective="multi:softprob", num_class=len(np.unique(y_enc)),
        eval_metric="mlogloss", tree_method="hist",
        random_state=42, reg_alpha=0.5, reg_lambda=2.0),
    param_distributions={
        "n_estimators"     : [100, 150, 200, 250],
        "max_depth"        : [3, 4, 5, 6],
        "learning_rate"    : [0.03, 0.05, 0.07, 0.1],
        "subsample"        : [0.7, 0.8, 0.9],
        "colsample_bytree" : [0.7, 0.8, 0.9],
        "min_child_weight" : [1, 3, 5],
    },
    n_iter      = N_ITER,
    cv          = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42),
    scoring     = "accuracy",
    random_state= 42,
    n_jobs      = -1,
    verbose     = 2)

t_s = time.time()
search.fit(X_tr_s, y_train, sample_weight=sw)
ts(f"✅ Done in {time.time()-t_s:.1f}s | Best CV: {search.best_score_:.4f}")

print("\n  Best hyperparameters:")
for k, v in search.best_params_.items():
    print(f"    {k}: {v}")
best_model = search.best_estimator_

# ================================================================
# STAGE 6 — EVALUATE + SAVE
# ================================================================
section(6, "Evaluation and save")
y_pred   = best_model.predict(X_te_s)
accuracy = accuracy_score(y_test, y_pred)
ts(f"🎯 Test Accuracy: {accuracy:.2%}")
print()
print(classification_report(y_test, y_pred,
                            target_names=le.classes_, digits=3))

# --- confusion matrix ---
cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=le.classes_).plot(
    cmap="Blues", values_format="d", ax=ax)
ax.set_title(f"Confusion Matrix — Test Accuracy: {accuracy:.2%}", fontsize=13)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_PATH, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight"); plt.close()
ts(f"Confusion matrix → {cm_path}")

# --- feature importance ---
fi = (pd.Series(best_model.feature_importances_, index=X.columns)
        .sort_values(ascending=False).head(20))
fig, ax = plt.subplots(figsize=(10, 6))
fi[::-1].plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 20 Feature Importances"); ax.set_xlabel("Importance")
plt.tight_layout()
fi_path = os.path.join(OUTPUT_PATH, "feature_importance.png")
plt.savefig(fi_path, dpi=150, bbox_inches="tight"); plt.close()
ts(f"Feature importance → {fi_path}")

# --- save model artefacts ---
saves = {
    "activity_model_realtime_new.pkl"  : best_model,
    "scaler_realtime.pkl"          : scaler,
    "selected_features_realtime.pkl" : X.columns.tolist(),
    "correlated_features_realtime.pkl": to_drop,
    "label_encoder_realtime.pkl"   : le,
    "window_size_realtime.pkl"     : WINDOW_SIZE,
}
for fname, obj in saves.items():
    joblib.dump(obj, os.path.join(OUTPUT_PATH, fname))
    ts(f"  Saved {fname}")

total = time.time() - t_start
print(f"""
{'='*62}
  TRAINING COMPLETE
{'='*62}
  Accuracy       : {accuracy:.2%}
  Features used  : {X.shape[1]}
  Window size    : {WINDOW_SIZE} rows ({WINDOW_SIZE*0.05:.1f}s)
  Classes        : {', '.join(le.classes_)}
  Total runtime  : {total:.0f}s ({total/60:.1f} min)
{'='*62}
""")