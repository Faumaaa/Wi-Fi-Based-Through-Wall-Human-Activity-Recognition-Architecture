"""
realtime_predict.py
Two-Stage Fusion: ML Model + Physics Rule Engine

THRESHOLDS RECALIBRATED FROM ACTUAL CSV DATA:
  kurtosis_value   : 2.4 – 13.0  (never reaches old threshold of 10 for walking)
  peak_to_avg_ratio: 2.9 –  5.7
  cadence_strength : 0.0 –  0.57
  energy           : 0.0007 – 0.08
    snr_db           : 12.5 – 31.6  (31+ = no target)
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ───────────────────────── Paths ──────────────────────────────────────────────
MODEL_PATH    = r"D:\FYP\activity_model_realtime_new.pkl"
SCALER_PATH   = r"D:\FYP\scaler_realtime.pkl"
FEATURES_PATH = r"D:\FYP\selected_features_realtime.pkl"
ENCODER_PATH  = r"D:\FYP\label_encoder_realtime.pkl"

CSV_FILE  = r"D:\FYP\realtime_features.csv"
FLAG_FILE = r"D:\FYP\data_ready.flag"

# ───────────────────────── Config ─────────────────────────────────────────────
WINDOW_SIZE          = 50
POLL_INTERVAL        = 1.0
SG_WINDOW            = 7
SG_POLYORDER         = 2
LF_BINS              = list(range(1, 9))
MODEL_WEIGHT         = 0.45
RULE_WEIGHT          = 0.55
MODEL_CONF_THRESHOLD = 0.35

# ───────────────────────── Stability / Hysteresis Config ──────────────────────
SWITCH_CONFIRM_COUNT        = 3    # frames current must lose before switching
SWITCH_CONFIRM_COUNT_FALL   = 6    # falling needs more evidence to trigger
SWITCH_MIN_MARGIN           = 0.08 # winner must beat current by at least this
SWITCH_MIN_MARGIN_FROM_FALL = 0.12 # harder to escape fall once confirmed

# ─── RECALIBRATED THRESHOLDS (from your actual USRP CSV data) ─────────────────
TH_KURTOSIS_ACTIVE  = 3.8   # kurtosis > this → active (walking/running)
TH_PEAK_AVG_ACTIVE  = 4.8   # PAR > this → active
TH_CADENCE_ACTIVE   = 0.06  # cadence_strength > this → active

TH_SNR_NO_TARGET    = 28.0  # SNR > this + very low energy → no target detected

TH_KURTOSIS_RUNNING = 3.6   # within active: kurtosis > this hints at running
TH_CADENCE_RUNNING  = 0.25  # cadence_strength > this → running
TH_CADENCE_PER_RUN  = 25.0  # cadence_period < this → running (fast steps)

# ─── GHOST ROW FILTER ─────────────────────────────────────────────────────────
GHOST_SNR_THRESHOLD      = 26.0
GHOST_CADENCE_THRESHOLD  = 0.005

# ───────────────────────── Load Artifacts ─────────────────────────────────────
print("Loading model artifacts...")
model    = joblib.load(MODEL_PATH)
scaler   = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)
encoder  = joblib.load(ENCODER_PATH)
CLASSES  = list(encoder.classes_)
print(f"Classes : {CLASSES}")
print(f"Features: {len(features)}")
print(f"CSV path: {CSV_FILE}\n")

_last_csv_hash = None


# ───────────────────────── Helpers ────────────────────────────────────────────
def gcol(df, name):
    return float(df[name].mean()) if name in df.columns else 0.0

def gcol_std(df, name):
    return float(df[name].std()) if name in df.columns else 0.0

def gcol_last(df, name, n=3):
    return float(df[name].tail(n).mean()) if name in df.columns else 0.0

def normalize(scores):
    scores = {k: max(0.0, v) for k, v in scores.items()}
    total  = sum(scores.values()) + 1e-10
    return {k: v / total for k, v in scores.items()}

def find_class(label):
    for c in CLASSES:
        if label.lower() in c.lower() or c.lower() in label.lower():
            return c
    return None

def add_score(scores, label, value):
    c = find_class(label)
    if c:
        scores[c] += max(0.0, value)
    return scores


# ───────────────────────── Stage 1: ML Feature Extraction ─────────────────────
def extract_window(df):
    agg = {}
    for col in df.columns:
        sig = df[col].values.astype(float)
        win = SG_WINDOW if len(sig) >= SG_WINDOW else \
              (len(sig) if len(sig) % 2 == 1 else len(sig) - 1)
        if win < 3:
            s = sig
        else:
            s = savgol_filter(sig, win, SG_POLYORDER)

        col_mean   = np.mean(s)
        col_std    = np.std(s)
        col_iqr    = np.percentile(s, 75) - np.percentile(s, 25)
        col_energy = np.sum(s ** 2)
        ps         = pd.Series(s)

        agg[col + "_mean"]          = col_mean
        agg[col + "_std"]           = col_std
        agg[col + "_var"]           = np.var(s)
        agg[col + "_median"]        = np.median(s)
        agg[col + "_max"]           = np.max(s)
        agg[col + "_min"]           = np.min(s)
        agg[col + "_range"]         = np.ptp(s)
        agg[col + "_skew"]          = float(ps.skew())
        agg[col + "_kurt"]          = float(min(ps.kurt(), 20))
        agg[col + "_iqr"]           = col_iqr
        agg[col + "_energy"]        = col_energy
        agg[col + "_rms"]           = np.sqrt(np.mean(s ** 2))
        fft_v = np.abs(np.fft.fft(s))
        agg[col + "_fft_mean"]      = np.mean(fft_v)
        agg[col + "_fft_std"]       = np.std(fft_v)
        agg[col + "_fft_max"]       = np.max(fft_v)
        agg[col + "_kurt_excess"]   = agg[col + "_kurt"] - 3
        agg[col + "_iqr_x_energy"]  = col_iqr * col_energy
        agg[col + "_stability"]     = col_std / (abs(col_mean) + 1e-10)
        for b in LF_BINS:
            agg[f"{col}_lf_fft_bin{b}"] = float(fft_v[b]) if b < len(fft_v) else 0.0
        segs      = np.array_split(s, 5)
        seg_stds  = [np.std(x)  for x in segs]
        seg_means = [np.mean(x) for x in segs]
        agg[col + "_var_of_std"]    = np.var(seg_stds)
        agg[col + "_range_of_mean"] = np.ptp(seg_means)
        agg[col + "_cv_of_std"]     = np.std(seg_stds) / (np.mean(seg_stds) + 1e-10)
    return agg


# ───────────────────────── Stage 2: Rule Engine ───────────────────────────────
def physics_scores(df):
    """
    Rule engine using recalibrated thresholds from actual CSV data.

    STEP 1: No-target check (SNR spike)
    STEP 2: Active vs Calm separation
    STEP 3: Calm → Sitting vs Standing
    STEP 4: Active → Walking vs Running
    STEP 5: Falling override — physics spike detector only
    """
    kurtosis      = gcol(df, 'kurtosis_value')
    peak_avg      = gcol(df, 'peak_to_avg_ratio')
    cadence_str   = gcol(df, 'cadence_strength')
    cadence_per   = gcol(df, 'cadence_period')
    energy        = gcol(df, 'energy')
    snr           = gcol(df, 'snr_db')
    zc_std        = gcol_std(df, 'zero_crossings')

    peak_recent   = gcol_last(df, 'peak_to_avg_ratio', 3)
    energy_recent = gcol_last(df, 'energy', 3)

    print(f"  [DATA] kurt={kurtosis:.1f}  PAR={peak_avg:.2f}  "
          f"cadStr={cadence_str:.3f}  cadPer={cadence_per:.0f}  "
          f"E={energy:.5f}  SNR={snr:.1f}  ZCstd={zc_std:.1f}")

    scores = {c: 0.0 for c in CLASSES}

    # ── STEP 1: No target? ────────────────────────────────────────────────────
    no_target = (snr > TH_SNR_NO_TARGET and energy < 0.00070)
    if no_target:
        scores = add_score(scores, 'sitting',  0.80)
        scores = add_score(scores, 'standing', 0.20)
        return normalize(scores)

    # ── STEP 2: Active or Calm? ───────────────────────────────────────────────
    is_active = (kurtosis    > TH_KURTOSIS_ACTIVE or
                 peak_avg    > TH_PEAK_AVG_ACTIVE  or
                 cadence_str > TH_CADENCE_ACTIVE)

    # ── STEP 3: Calm → Sitting vs Standing ───────────────────────────────────
    if not is_active:
        if cadence_str < 0.035 and zc_std < 50:
            # Very little motion → sitting
            scores = add_score(scores, 'sitting',  0.75)
            scores = add_score(scores, 'standing', 0.25)
        elif cadence_str < 0.035 and zc_std >= 50:
            # Low cadence but signal variation → standing
            scores = add_score(scores, 'standing', 0.65)
            scores = add_score(scores, 'sitting',  0.35)
        else:
            # cadence_str 0.035–0.06 → micro body sway → standing
            scores = add_score(scores, 'standing', 0.75)
            scores = add_score(scores, 'sitting',  0.25)

    # ── STEP 4: Active → Walking vs Running ──────────────────────────────────
    else:
        is_running = (kurtosis    > TH_KURTOSIS_RUNNING or
                      cadence_str > TH_CADENCE_RUNNING   or
                      (cadence_per > 0 and cadence_per < TH_CADENCE_PER_RUN))

        if is_running:
            run_votes = sum([
                kurtosis    > TH_KURTOSIS_RUNNING,
                cadence_str > TH_CADENCE_RUNNING,
                cadence_per > 0 and cadence_per < TH_CADENCE_PER_RUN,
            ])
            run_conf = 0.60 + 0.13 * run_votes  # 0.60 / 0.73 / 0.86
            scores = add_score(scores, 'running', run_conf)
            scores = add_score(scores, 'walking', 1.0 - run_conf)
        else:
            scores = add_score(scores, 'walking', 0.80)
            scores = add_score(scores, 'running', 0.20)

    # ── STEP 5: Falling — physics spike detector ──────────────────────────────
    # A real fall = sudden burst in peak_to_avg_ratio AND energy,
    # with low cadence (person not actively walking/running into it).
    peak_spike   = peak_recent / (peak_avg + 1e-10)
    energy_spike = energy_recent / (energy + 1e-10)

    if peak_spike > 2.0 and energy_spike > 1.5 and cadence_str < 0.10:
        fall_score = min(0.70, 0.25 * peak_spike)
        fall_cls   = find_class('falling')
        if fall_cls:
            for c in scores:
                scores[c] *= (1.0 - fall_score)
            scores[fall_cls] = scores.get(fall_cls, 0.0) + fall_score
            print(f"  [⚠️ FALL DETECTED] peak_spike={peak_spike:.1f}x  "
                  f"energy_spike={energy_spike:.1f}x")

    return normalize(scores)


# ───────────────────────── Stage 3: Fusion ────────────────────────────────────
def fuse(model_probs, rule_scores, model_conf):
    if model_conf < MODEL_CONF_THRESHOLD:
        # Low-confidence ML → lean more on rules
        mw = MODEL_WEIGHT * (model_conf / MODEL_CONF_THRESHOLD)
        rw = 1.0 - mw
    else:
        mw = MODEL_WEIGHT
        rw = RULE_WEIGHT

    fused = {}
    for i, cls in enumerate(CLASSES):
        mp = float(model_probs[i]) if i < len(model_probs) else 0.0
        rp = rule_scores.get(cls, 0.0)
        fused[cls] = mw * mp + rw * rp

    return normalize(fused), mw, rw


# ───────────────────────── Stability Gate ─────────────────────────────────────
class StablePredictor:
    """
    Switches only when the CURRENT activity has been convincingly losing
    for SWITCH_CONFIRM_COUNT consecutive frames.

    Key improvement over simple challenger tracking:
    - Tracks losing streak of CURRENT (not winning streak of any single challenger)
    - So walking→running alternating frames both count against 'standing'
    - Winner = activity with highest accumulated score over the losing streak
    - 'Too close' frames (margin < threshold) do NOT count and do NOT reset streak
    - Falling requires 6 frames to trigger, 12% margin to escape
    """
    def __init__(self):
        self.current_label   = None
        self.current_conf    = 0.0
        self.losing_streak   = 0
        self.challenger_pool = {}   # accumulated scores while current is losing

    def update(self, fused: dict) -> tuple:
        best_label = max(fused, key=fused.get)
        best_conf  = fused[best_label]

        # ── First prediction ever ─────────────────────────────────────────────
        if self.current_label is None:
            self.current_label = best_label
            self.current_conf  = best_conf
            return best_label, best_conf, True

        current_score = fused.get(self.current_label, 0.0)

        # ── Current activity still on top → reset streak ──────────────────────
        if best_label == self.current_label:
            self.current_conf    = best_conf
            self.losing_streak   = 0
            self.challenger_pool = {}
            return self.current_label, self.current_conf, False

        # ── Current is NOT on top — check margin ──────────────────────────────
        margin = best_conf - current_score

        # Use stricter margin to escape a fall lock
        min_margin = SWITCH_MIN_MARGIN_FROM_FALL \
                     if self.current_label == find_class('falling') \
                     else SWITCH_MIN_MARGIN

        if margin < min_margin:
            # Too close — don't count this frame, but also DON'T reset the streak
            # (a single close frame shouldn't forgive a genuine losing run)
            print(f"  [STABLE] '{best_label}'({best_conf:.2f}) vs "
                  f"'{self.current_label}'({current_score:.2f}) "
                  f"margin={margin:.2f} < {min_margin:.2f} — too close, skip.")
            return self.current_label, self.current_conf, False

        # ── Current convincingly beaten — accumulate evidence ─────────────────
        self.losing_streak += 1
        for cls, score in fused.items():
            if cls != self.current_label:
                self.challenger_pool[cls] = \
                    self.challenger_pool.get(cls, 0.0) + score

        # Falling needs more frames to trigger (rare, high-stakes event)
        confirm_needed = SWITCH_CONFIRM_COUNT_FALL \
                         if find_class('falling') == best_label \
                         else SWITCH_CONFIRM_COUNT

        top_challenger = max(self.challenger_pool, key=self.challenger_pool.get)
        top_score      = self.challenger_pool[top_challenger]
        remaining      = confirm_needed - self.losing_streak

        print(f"  [STABLE] '{self.current_label}' losing "
              f"streak={self.losing_streak}/{confirm_needed}  "
              f"leader='{top_challenger}'(pool={top_score:.2f})"
              + (f"  — {remaining} more" if remaining > 0 else "  — SWITCHING!"))

        if self.losing_streak >= confirm_needed:
            new_label = max(self.challenger_pool, key=self.challenger_pool.get)
            new_conf  = fused.get(new_label, best_conf)
            self.current_label   = new_label
            self.current_conf    = new_conf
            self.losing_streak   = 0
            self.challenger_pool = {}
            return self.current_label, self.current_conf, True

        return self.current_label, self.current_conf, False


stable = StablePredictor()   # ← single instance, never duplicated


# ───────────────────────── Predict ────────────────────────────────────────────
def predict(df):
    window = df.tail(WINDOW_SIZE).copy()
    if len(window) < WINDOW_SIZE:
        print("[INFO] Not enough rows yet...")
        return

    # ── Clean ghost rows (no-target SNR spikes) before any processing ─────────
    ghost_mask = ((window['snr_db'] > GHOST_SNR_THRESHOLD) &
                  (window['cadence_strength'] < GHOST_CADENCE_THRESHOLD))
    n_ghosts = ghost_mask.sum()
    if n_ghosts > 0:
        print(f"  [CLEAN] Removed {n_ghosts} ghost rows (SNR spike, no target)")
        window.loc[ghost_mask] = np.nan
        window = window.interpolate(method='linear').ffill().bfill()

    # ── Stage 1: ML model ─────────────────────────────────────────────────────
    feat_dict   = extract_window(window)
    X           = pd.DataFrame([feat_dict])
    for col in features:
        if col not in X.columns:
            X[col] = 0.0
    X           = X[features]
    X_scaled    = scaler.transform(X)
    model_probs = model.predict_proba(X_scaled)[0]
    model_pred  = int(np.argmax(model_probs))
    model_label = encoder.inverse_transform([model_pred])[0]
    model_conf  = float(np.max(model_probs))

    # ── Stage 2: Rule engine ──────────────────────────────────────────────────
    rule_scores = physics_scores(window)

    # ── Stage 3: Fusion ───────────────────────────────────────────────────────
    fused, mw, rw = fuse(model_probs, rule_scores, model_conf)

    # ── Hard-block falling unless physics spike detector fired ────────────────
    # The ML model can output 'falling' from noisy features. We only allow it
    # when there is a real measured burst in peak_to_avg_ratio + energy.
    peak_avg      = gcol(window, 'peak_to_avg_ratio')
    energy        = gcol(window, 'energy')
    peak_recent   = gcol_last(window, 'peak_to_avg_ratio', 3)
    energy_recent = gcol_last(window, 'energy', 3)
    cadence_str   = gcol(window, 'cadence_strength')
    peak_spike    = peak_recent / (peak_avg + 1e-10)
    energy_spike  = energy_recent / (energy + 1e-10)
    physics_fall_fired = (peak_spike > 2.0 and
                          energy_spike > 1.5 and
                          cadence_str < 0.10)

    fall_cls = find_class('falling')
    if fall_cls and not physics_fall_fired:
        fused[fall_cls] = 0.0
        fused = normalize(fused)

    # ── Stability gate ────────────────────────────────────────────────────────
    display_label, display_conf, switched = stable.update(fused)

    # ── Output ────────────────────────────────────────────────────────────────
    switch_tag = " ◀ NEW" if switched else ""
    print(f"\n  ✅ ACTIVITY : {display_label.upper()}{switch_tag}  "
          f"({display_conf:.0%})")
    print("  ── All scores ──")
    for cls in sorted(fused, key=fused.get, reverse=True):
        marker = "◀" if cls == display_label else " "
        print(f"  {marker} {cls:<12} {fused[cls]:.2f}")


# ───────────────────────── CSV Helpers ────────────────────────────────────────
def safe_read_csv(path, retries=3, delay=0.2):
    for attempt in range(retries):
        try:
            df = pd.read_csv(path)
            if df.empty or df.shape[1] != 30:
                print(f"[WARN] CSV has {df.shape[1]} cols, expected 30.")
                return None
            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"[WARN] CSV read failed: {e}")
    return None


def csv_changed(path):
    global _last_csv_hash
    try:
        tail     = pd.read_csv(path).tail(3).to_csv()
        new_hash = hash(tail)
        if new_hash != _last_csv_hash:
            _last_csv_hash = new_hash
            return True
        return False
    except Exception:
        return False


# ───────────────────────── Main Loop ──────────────────────────────────────────
print("=" * 65)
print("  Two-Stage Fusion — 5 Class Radar Activity Recognition")
print("  Walking | Running | Sitting | Standing | Falling")
print(f"  Reading from: {CSV_FILE}")
print("=" * 65 + "\n")

last_flag_mod = 0.0
last_predict  = 0.0

while True:
    try:
        now = time.time()

        if now - last_predict < 1.0:
            time.sleep(0.1)
            continue

        flag_ok = os.path.exists(FLAG_FILE)
        csv_ok  = os.path.exists(CSV_FILE)

        if flag_ok and csv_ok:
            flag_mod = os.path.getmtime(FLAG_FILE)
            if flag_mod != last_flag_mod:
                last_flag_mod = flag_mod
                last_predict  = now

                if csv_changed(CSV_FILE):
                    print(f"\n[{time.strftime('%H:%M:%S')}] New data detected...")
                    df = safe_read_csv(CSV_FILE)
                    if df is not None and len(df) >= WINDOW_SIZE:
                        predict(df)
                    else:
                        print("[INFO] CSV not ready.")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Flag touched but CSV unchanged.")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Waiting —"
                  f" flag={'✅' if flag_ok else '❌'}"
                  f" csv={'✅' if csv_ok else '❌'}", end='\r')

    except KeyboardInterrupt:
        print("\nStopped.")
        break
    except Exception as e:
        print(f"[WARN] {e}")

    time.sleep(POLL_INTERVAL)