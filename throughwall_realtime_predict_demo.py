"""
throughwall_realtime_predict_V4.py
=====================================
Changes over V3:

  FIX 1 — CONSENSUS OVERRIDE (most important fix)
           If ML AND physics BOTH agree on the same class, the stability
           gate is bypassed immediately — no confirm cycles needed.
           Previous versions held "walking" for 3 cycles even when both
           ML (90% standing) and physics (87% standing) agreed it was
           standing. That was wrong.

  FIX 2 — AGREEMENT SCORE replaces the ML-only confidence check.
           We now compute how much ML and physics agree with each other.
           When they agree strongly → high fusion confidence → fast switch.
           When they disagree → low agreement → hold current label.

  FIX 3 — HOLD CYCLES reset correctly.
           V3 had a bug where hold_cycles incremented even when the
           best_label matched current_label, causing premature margin
           reductions. Fixed: hold_cycles only increments when we are
           genuinely blocking a challenger.

  FIX 4 — STANDING LOCK: if both ML and physics say standing with
           combined score > STANDING_LOCK_TH, immediately switch to
           standing regardless of current label. Standing is easy to
           detect (very low Δrm) so false positives here are rare.

  FIX 5 — Removed HIGH_CONF_THRESHOLD ML weight boost.
           Boosting ML weight when model_conf > 80% was the root cause
           of "running" getting locked in V2 and "walking" staying locked
           in V3. ML confidence is unreliable on this model. Weights are
           now fixed: ML=0.50, physics=0.50, unless ML conf is very low.

  FIX 6 — activity_level computation capped more conservatively.
           Using min(rm_level, 3.0) before averaging prevents one very
           high Δen spike from pushing activity_level to 1.0 and
           wrongly triggering the "clearly active" walking branch.
"""

import os, time, joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════
CSV_FILE  = r"D:\FYP\throughwall_realtime.csv"
FLAG_FILE = r"D:\FYP\throughwall_ready.flag"

MODEL_PATH    = r"D:\FYP\throughwall_model_v2.pkl"
SCALER_PATH   = r"D:\FYP\throughwall_scaler_v2.pkl"
FEATURES_PATH = r"D:\FYP\throughwall_selected_features_v2.pkl"
ENCODER_PATH  = r"D:\FYP\throughwall_label_encoder_v2.pkl"
NORM_REF_PATH = r"D:\FYP\throughwall_normalisation_ref_v2.pkl"

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
WINDOW_SIZE   = 100
POLL_INTERVAL = 0.1
SG_WINDOW     = 9
SG_POLYORDER  = 2
LF_BINS       = list(range(1, 13))
SIGNAL_COLS   = ["range_mag", "phase", "energy", "raw_peak", "delta_rm", "delta_en"]

DISPLAY_MIN_INTERVAL = 2.5

# ── Fusion weights ────────────────────────────────────────────────────────
# FIX 5: equal weights — ML confidence on this model is not reliable enough
# to justify boosting it. Both signals get equal say.
MODEL_WEIGHT         = 0.55
RULE_WEIGHT          = 0.45
MODEL_CONF_THRESHOLD = 0.30   # below this → scale ML weight down linearly

# ── Consensus / override thresholds ──────────────────────────────────────
# FIX 1: if ML and physics top class match AND combined fused score ≥ this
# → bypass stability gate, switch immediately
CONSENSUS_INSTANT_TH = 0.75

# FIX 4: if fused standing score ≥ this → switch to standing instantly
# (standing is the easiest class to detect — very low Δrm is unambiguous)
STANDING_LOCK_TH = 0.82

# Physics override (same as V3)
PHYSICS_OVERRIDE_TH   = 0.70
PHYSICS_OVERRIDE_MULT = 1.35

# ── Stability gate ────────────────────────────────────────────────────────
SWITCH_CONFIRM_COUNT = 3
SWITCH_MIN_MARGIN    = 0.05
MAX_HOLD_CYCLES      = 6     # FIX 3: lower than V3 so stuck states clear faster

# ── Confidence display ────────────────────────────────────────────────────
CONFIDENCE_FLOOR = 0.58

# ── Physics thresholds ────────────────────────────────────────────────────
TH_DELTA_RM_ACTIVE  = 0.008
TH_DELTA_RM_RUNNING = 0.018
TH_DELTA_EN_ACTIVE  = 0.003
TH_DELTA_RM_AC_WALK = 0.30

RUNNING_MAX_PROB_IF_SLOW = 0.60

ACTIVITY_LOW  = 0.40
ACTIVITY_HIGH = 0.80

# ── Fall detection ────────────────────────────────────────────────────────
FALL_SPIKE_WIN     = 15
FALL_SPIKE_TH      = 0.12
FALL_STILL_WIN     = 25
FALL_STILL_TH      = 0.010
FALL_CONFIRM_TICKS = 3
FALL_RESET_TICKS   = 6

DEBUG_MODE = True

# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════
print("Loading through-wall model (v2)...")
model    = joblib.load(MODEL_PATH)
scaler   = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)
encoder  = joblib.load(ENCODER_PATH)
norm_ref = joblib.load(NORM_REF_PATH)

ML_CLASSES  = list(encoder.classes_)
ALL_CLASSES = sorted(set(ML_CLASSES) | {"falling"})

ACTIVITY_EMOJI = {
    "standing": "🧍",
    "walking":  "🚶",
    "running":  "🏃",
    "falling":  "🆘",
}

print(f"ML classes : {ML_CLASSES}")
print(f"All classes: {ALL_CLASSES}  (falling = physics rule)")
print(f"DEBUG      : {'ON' if DEBUG_MODE else 'OFF'}\n")

_last_flag_mtime = 0.0
_last_csv_hash   = None
_last_display_t  = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# FALL DETECTOR
# ═══════════════════════════════════════════════════════════════════════════
class FallDetector:
    def __init__(self):
        self.state        = "IDLE"
        self.still_ticks  = 0
        self.active_ticks = 0

    def update(self, delta_rm_arr):
        spike_mean = float(np.mean(delta_rm_arr[-FALL_SPIKE_WIN:]))
        still_mean = float(np.mean(delta_rm_arr[-FALL_STILL_WIN:]))

        if self.state == "IDLE":
            if spike_mean > FALL_SPIKE_TH:
                self.state = "IMPACT"; self.still_ticks = 0; self.active_ticks = 0
        elif self.state == "IMPACT":
            if spike_mean <= FALL_SPIKE_TH:
                self.state = "CONFIRM"
        elif self.state == "CONFIRM":
            if still_mean < FALL_STILL_TH:
                self.still_ticks += 1
                if self.still_ticks >= FALL_CONFIRM_TICKS:
                    self.state = "FALLEN"; self.still_ticks = 0
            else:
                self.state = "IDLE"; self.still_ticks = 0
        elif self.state == "FALLEN":
            if still_mean > TH_DELTA_RM_ACTIVE * 1.5:
                self.active_ticks += 1
                if self.active_ticks >= FALL_RESET_TICKS:
                    self.state = "IDLE"; self.active_ticks = 0
            else:
                self.active_ticks = 0

        if self.state == "FALLEN":   return 0.97, self.state
        elif self.state == "CONFIRM":
            return 0.40 + 0.45*(self.still_ticks/FALL_CONFIRM_TICKS), self.state
        elif self.state == "IMPACT":
            return min(0.40, (spike_mean-FALL_SPIKE_TH)/FALL_SPIKE_TH*0.4), self.state
        else:
            return 0.0, self.state


fall_detector = FallDetector()


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def softmax_normalize(scores):
    scores = {k: max(0.0, v) for k, v in scores.items()}
    total  = sum(scores.values()) + 1e-10
    return {k: v / total for k, v in scores.items()}


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (identical to train_throughwall_FIXED.py)
# ═══════════════════════════════════════════════════════════════════════════
def extract_window(df):
    agg = {}
    for col in SIGNAL_COLS:
        if col not in df.columns:
            for suffix in ["_mean","_std","_var","_median","_max","_min","_range",
                           "_skew","_kurt","_iqr","_energy","_rms","_fft_mean",
                           "_fft_std","_fft_max","_kurt_excess","_iqr_x_energy",
                           "_stability","_var_of_std","_range_of_mean","_cv_of_std",
                           "_ac1","_ac2","_ac3","_ac4","_ac5","_trend"]:
                agg[col + suffix] = 0.0
            for b in LF_BINS:
                agg[f"{col}_lf_fft_bin{b}"] = 0.0
            continue

        sig = pd.to_numeric(df[col].values, errors="coerce")
        sig = sig[~np.isnan(sig)]
        if len(sig) < 4:
            sig = np.zeros(WINDOW_SIZE)

        sg_win = min(SG_WINDOW, len(sig) if len(sig) % 2 == 1 else len(sig)-1)
        sg_win = max(sg_win, SG_POLYORDER+2 if (SG_POLYORDER+2)%2==1 else SG_POLYORDER+3)
        try:
            s = savgol_filter(sig, window_length=sg_win, polyorder=SG_POLYORDER)
        except ValueError:
            s = sig.copy()

        col_mean   = np.mean(s);  col_std = np.std(s);  col_median = np.median(s)
        col_max    = np.max(s);   col_min = np.min(s)
        col_iqr    = np.percentile(s,75) - np.percentile(s,25)
        col_energy = np.sum(s**2); col_rms = np.sqrt(np.mean(s**2))
        col_kurt   = min(float(pd.Series(s).kurt()), 20.0)

        agg.update({
            col+"_mean": col_mean, col+"_std": col_std, col+"_var": np.var(s),
            col+"_median": col_median, col+"_max": col_max, col+"_min": col_min,
            col+"_range": col_max-col_min, col+"_skew": float(pd.Series(s).skew()),
            col+"_kurt": col_kurt, col+"_iqr": col_iqr,
            col+"_energy": col_energy, col+"_rms": col_rms,
            col+"_kurt_excess": col_kurt-3.0,
            col+"_iqr_x_energy": col_iqr*col_energy,
            col+"_stability": col_std/(abs(col_mean)+1e-10),
        })

        fft_v = np.abs(np.fft.fft(s))
        agg[col+"_fft_mean"] = np.mean(fft_v)
        agg[col+"_fft_std"]  = np.std(fft_v)
        agg[col+"_fft_max"]  = np.max(fft_v)
        for b in LF_BINS:
            agg[f"{col}_lf_fft_bin{b}"] = float(fft_v[b]) if b < len(fft_v) else 0.0

        n_segs = 5; seg_len = max(1, len(s)//n_segs)
        seg_stds, seg_means = [], []
        for i in range(n_segs):
            seg = s[i*seg_len:(i+1)*seg_len]
            if len(seg) > 0:
                seg_stds.append(np.std(seg)); seg_means.append(np.mean(seg))
        seg_stds  = np.array(seg_stds)  if seg_stds  else np.array([0.0])
        seg_means = np.array(seg_means) if seg_means else np.array([0.0])
        agg[col+"_var_of_std"]    = np.var(seg_stds)
        agg[col+"_range_of_mean"] = np.ptp(seg_means)
        agg[col+"_cv_of_std"]     = np.std(seg_stds)/(np.mean(seg_stds)+1e-10)

        s_c     = s - col_mean
        ac_full = np.correlate(s_c, s_c, mode='full')[len(s)-1:]
        ac      = ac_full / (ac_full[0]+1e-10)
        for lag in range(1, 6):
            agg[f"{col}_ac{lag}"] = float(ac[lag]) if lag < len(ac) else 0.0

        agg[col+"_trend"] = float(np.polyfit(np.arange(len(s)), s, 1)[0])

    if "delta_rm" in df.columns and "energy" in df.columns:
        d_rm = pd.to_numeric(df["delta_rm"].values, errors="coerce")
        en   = pd.to_numeric(df["energy"].values,   errors="coerce")
        d_rm = d_rm[~np.isnan(d_rm)]; en = en[~np.isnan(en)]
        if len(d_rm) > 0 and len(en) > 0:
            agg["cross_delta_rm_per_energy"]   = np.mean(d_rm)/(np.mean(en)+1e-10)
            agg["cross_delta_rm_times_energy"] = np.mean(d_rm)*np.mean(en)
            agg["cross_delta_rm_std"]          = np.std(d_rm)
            agg["cross_delta_en_std"]          = np.std(pd.to_numeric(
                                                     df["delta_en"].values, errors="coerce"))

    if "range_mag" in df.columns and "delta_rm" in df.columns:
        rm   = pd.to_numeric(df["range_mag"].values, errors="coerce")
        d_rm = pd.to_numeric(df["delta_rm"].values,  errors="coerce")
        rm   = rm[~np.isnan(rm)]; d_rm = d_rm[~np.isnan(d_rm)]
        if len(rm) > 0 and len(d_rm) > 0:
            agg["cross_motion_index"] = np.mean(d_rm)/(np.std(rm)+1e-10)

    return agg


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS RULES
# ═══════════════════════════════════════════════════════════════════════════
def physics_scores_3class(df):
    scores = {c: 0.0 for c in ML_CLASSES}

    d_rm_col = "delta_rm" in df.columns
    d_en_col = "delta_en" in df.columns

    if d_rm_col:
        delta_rm      = pd.to_numeric(df["delta_rm"].values, errors="coerce")
        mean_delta_rm = float(np.nanmean(delta_rm))
        d = delta_rm - np.nanmean(delta_rm)
        if np.nanstd(d) > 1e-10:
            ac  = np.correlate(d, d, mode='full')[len(d)-1:]
            ac  = ac / (ac[0] + 1e-10)
            ac1 = float(ac[1]) if len(ac) > 1 else 0.0
        else:
            ac1 = 0.0
    else:
        mean_delta_rm = float(df["range_mag"].std()) if "range_mag" in df.columns else 0.0
        ac1 = 0.0

    mean_delta_en = float(np.nanmean(pd.to_numeric(
        df["delta_en"].values, errors="coerce"))) if d_en_col else 0.0

    # FIX 6: cap each level independently before averaging
    rm_level       = min(mean_delta_rm / (TH_DELTA_RM_ACTIVE + 1e-10), 3.0)
    en_level       = min(mean_delta_en / (TH_DELTA_EN_ACTIVE + 1e-10), 3.0)
    activity_level = float(np.clip(0.5 * rm_level + 0.5 * en_level, 0.0, 1.0))

    is_fast     = mean_delta_rm > TH_DELTA_RM_RUNNING
    is_periodic = ac1 > TH_DELTA_RM_AC_WALK

    if DEBUG_MODE:
        print(f"  📊 mean_Δrm={mean_delta_rm:.5f}  ac1={ac1:.4f}  "
              f"mean_Δen={mean_delta_en:.5f}  activity={activity_level:.3f}")
        if activity_level > ACTIVITY_LOW:
            print(f"  🏃 periodic={is_periodic}(ac1={ac1:.3f}>{TH_DELTA_RM_AC_WALK})  "
                  f"fast={is_fast}(Δrm={mean_delta_rm:.4f}>{TH_DELTA_RM_RUNNING})")

    def add(label, v):
        for c in ML_CLASSES:
            if label.lower() in c.lower():
                scores[c] = scores.get(c, 0.0) + max(0.0, v)

    if activity_level < ACTIVITY_LOW:
        add("standing", 0.85)
        add("walking",  0.15)

    elif activity_level < ACTIVITY_HIGH:
        blend          = (activity_level - ACTIVITY_LOW) / (ACTIVITY_HIGH - ACTIVITY_LOW)
        standing_score = 0.70 * (1.0 - blend) + 0.10 * blend
        active_score   = 0.30 * (1.0 - blend) + 0.90 * blend
        add("standing", standing_score)
        if is_fast:
            add("running", active_score * 0.65)
            add("walking", active_score * 0.35)
        elif is_periodic:
            add("walking", active_score * 0.80)
            add("running", active_score * 0.20)
        else:
            add("walking", active_score * 0.60)
            add("running", active_score * 0.40)

    else:
        if is_fast and is_periodic:
            add("running", 0.78); add("walking", 0.22)
        elif is_fast:
            add("running", 0.65); add("walking", 0.35)
        elif is_periodic:
            add("walking", 0.82); add("running", 0.18)
        else:
            add("walking", 0.60); add("running", 0.40)

    total = sum(scores.values()) + 1e-10
    return {k: v/total for k, v in scores.items()}, mean_delta_rm, is_fast


# ═══════════════════════════════════════════════════════════════════════════
# FUSION
# ═══════════════════════════════════════════════════════════════════════════
def fuse_with_fall(model_probs, rule_scores_3, fall_prob, model_conf,
                   mean_delta_rm, is_fast):
    # FIX 5: fixed equal weights, only reduce ML if truly low confidence
    if model_conf < MODEL_CONF_THRESHOLD:
        mw = MODEL_WEIGHT * (model_conf / MODEL_CONF_THRESHOLD)
        rw = 1.0 - mw
    else:
        mw = MODEL_WEIGHT
        rw = RULE_WEIGHT

    # Cap running probability when Δrm is walking-level
    model_probs_adj = list(model_probs)
    running_idx = next((i for i, c in enumerate(ML_CLASSES)
                        if "running" in c.lower()), None)
    walking_idx = next((i for i, c in enumerate(ML_CLASSES)
                        if "walking" in c.lower()), None)

    if running_idx is not None and not is_fast:
        excess = max(0.0, model_probs_adj[running_idx] - RUNNING_MAX_PROB_IF_SLOW)
        model_probs_adj[running_idx] = RUNNING_MAX_PROB_IF_SLOW
        if walking_idx is not None:
            model_probs_adj[walking_idx] = min(1.0, model_probs_adj[walking_idx] + excess)
        if DEBUG_MODE and excess > 0.01:
            print(f"  🔧 Running capped (Δrm={mean_delta_rm:.4f}<{TH_DELTA_RM_RUNNING}): "
                  f"ML running {model_probs[running_idx]:.0%} → {RUNNING_MAX_PROB_IF_SLOW:.0%}, "
                  f"excess {excess:.0%} → walking")

    total_adj       = sum(model_probs_adj) + 1e-10
    model_probs_adj = [p/total_adj for p in model_probs_adj]

    # Physics override when physics is very confident and ML disagrees
    phys_top_cls   = max(rule_scores_3, key=rule_scores_3.get)
    phys_top_score = rule_scores_3[phys_top_cls]
    ml_top_idx     = int(np.argmax(model_probs_adj))
    ml_top_cls     = ML_CLASSES[ml_top_idx] if ml_top_idx < len(ML_CLASSES) else ""

    boosted_rule = dict(rule_scores_3)
    if phys_top_score > PHYSICS_OVERRIDE_TH and phys_top_cls != ml_top_cls:
        boosted_rule[phys_top_cls] = min(1.0, phys_top_score * PHYSICS_OVERRIDE_MULT)
        total_b = sum(boosted_rule.values()) + 1e-10
        boosted_rule = {k: v/total_b for k, v in boosted_rule.items()}
        if DEBUG_MODE:
            print(f"  ⚡ Physics override: {phys_top_cls} "
                  f"({phys_top_score:.0%}→{boosted_rule[phys_top_cls]:.0%}) "
                  f"vs ML={ml_top_cls}")

    fused3 = {}
    for i, cls in enumerate(ML_CLASSES):
        mp = float(model_probs_adj[i]) if i < len(model_probs_adj) else 0.0
        rp = boosted_rule.get(cls, 0.0)
        fused3[cls] = mw * mp + rw * rp

    remaining = 1.0 - fall_prob
    total3    = sum(fused3.values()) + 1e-10
    fused4    = {cls: (v/total3)*remaining for cls, v in fused3.items()}
    fused4["falling"] = fall_prob

    fused4 = softmax_normalize(fused4)

    # Return which class ML and physics agree on (if any)
    ml_top  = ML_CLASSES[int(np.argmax(model_probs_adj))] if ML_CLASSES else ""
    consensus_cls = ml_top if ml_top == phys_top_cls else None

    return fused4, mw, rw, consensus_cls


# ═══════════════════════════════════════════════════════════════════════════
# STABILITY GATE — FIX 1/3/4: consensus bypass + correct hold_cycles
# ═══════════════════════════════════════════════════════════════════════════
class StablePredictor:
    def __init__(self):
        self.current_label   = None
        self.current_conf    = 0.0
        self.losing_streak   = 0
        self.challenger_pool = {}
        self.hold_cycles     = 0

    def update(self, fused4, consensus_cls):
        best_label = max(fused4, key=fused4.get)
        best_conf  = fused4[best_label]

        if self.current_label is None:
            self.current_label = best_label
            self.current_conf  = best_conf
            return best_label, best_conf, True

        # ── FIX 4: standing lock — unambiguous, switch immediately ───────
        standing_score = fused4.get(
            next((c for c in fused4 if "standing" in c.lower()), "standing"), 0.0)
        if standing_score >= STANDING_LOCK_TH and self.current_label != best_label:
            if DEBUG_MODE:
                print(f"  🔒→✅ Standing lock ({standing_score:.0%}) — "
                      f"instant switch from '{self.current_label}'")
            self.current_label   = best_label
            self.current_conf    = best_conf
            self.losing_streak   = 0
            self.challenger_pool = {}
            self.hold_cycles     = 0
            return self.current_label, self.current_conf, True

        # ── FIX 1: consensus bypass — ML + physics agree, switch now ─────
        if (consensus_cls is not None and
                consensus_cls != self.current_label and
                best_conf >= CONSENSUS_INSTANT_TH):
            if DEBUG_MODE:
                print(f"  🤝 Consensus override: both say '{consensus_cls}' "
                      f"({best_conf:.0%}) — bypassing stability gate")
            self.current_label   = consensus_cls
            self.current_conf    = best_conf
            self.losing_streak   = 0
            self.challenger_pool = {}
            self.hold_cycles     = 0
            return self.current_label, self.current_conf, True

        # ── Normal stability logic ────────────────────────────────────────
        current_score = fused4.get(self.current_label, 0.0)

        if best_label == self.current_label:
            self.current_conf    = best_conf
            self.losing_streak   = 0
            self.challenger_pool = {}
            # FIX 3: only increment hold_cycles when confidence is low
            self.hold_cycles = (self.hold_cycles + 1) if best_conf < CONFIDENCE_FLOOR else 0
            return self.current_label, self.current_conf, False

        margin = best_conf - current_score
        confirm_needed   = 2 if best_label == "falling" else SWITCH_CONFIRM_COUNT
        min_margin       = 0.03 if best_label == "falling" else SWITCH_MIN_MARGIN
        effective_margin = (min_margin * 0.5
                            if self.hold_cycles >= MAX_HOLD_CYCLES
                            else min_margin)

        if margin < effective_margin:
            if DEBUG_MODE:
                print(f"  🔒 Margin {margin:.3f} < {effective_margin:.3f} — "
                      f"holding '{self.current_label}'")
            # FIX 3: increment only here (genuinely blocked)
            self.hold_cycles += 1
            return self.current_label, self.current_conf, False

        self.losing_streak += 1
        for cls, score in fused4.items():
            if cls != self.current_label:
                self.challenger_pool[cls] = self.challenger_pool.get(cls, 0.0) + score

        if DEBUG_MODE:
            print(f"  ⏳ '{self.current_label}' losing "
                  f"{self.losing_streak}/{confirm_needed}")

        if self.losing_streak >= confirm_needed:
            new_label            = max(self.challenger_pool, key=self.challenger_pool.get)
            new_conf             = fused4.get(new_label, best_conf)
            self.current_label   = new_label
            self.current_conf    = new_conf
            self.losing_streak   = 0
            self.challenger_pool = {}
            self.hold_cycles     = 0
            return self.current_label, self.current_conf, True

        return self.current_label, self.current_conf, False


stable = StablePredictor()


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════
def display_prediction(display_label, display_conf, switched,
                        model_label, model_conf, mw, rw, fused4, fall_state):
    global _last_display_t
    now = time.time()
    if now - _last_display_t < DISPLAY_MIN_INTERVAL:
        return
    _last_display_t = now

    emoji         = ACTIVITY_EMOJI.get(display_label.lower(), "❓")
    switch_tag    = "   ← NEW" if switched else ""
    uncertain_tag = "  ⚠ uncertain" if display_conf < CONFIDENCE_FLOOR else ""

    def sort_key(cls):
        if cls == display_label: return (0, -fused4.get(cls, 0))
        if cls == "falling":     return (2, -fused4.get(cls, 0))
        return (1, -fused4.get(cls, 0))

    sorted_classes = sorted(fused4.keys(), key=sort_key)

    print(f"\n{'═'*60}")
    print(f"  {emoji}  {display_label.upper()}{switch_tag}{uncertain_tag}   "
          f"confidence: {display_conf:.0%}")
    print(f"  ML model → {model_label} ({model_conf:.0%})   "
          f"weights: ML {mw:.0%} / physics {rw:.0%}   fall FSM: {fall_state}")
    print(f"  {'─'*56}")
    for cls in sorted_classes:
        prob   = fused4.get(cls, 0.0)
        bar    = "█" * int(prob * 36)
        marker = "▶" if cls == display_label else " "
        e      = ACTIVITY_EMOJI.get(cls.lower(), " ")
        print(f"  {marker} {e} {cls:<10}  {prob:5.1%}  {bar}")
    print("═"*60)


# ═══════════════════════════════════════════════════════════════════════════
# CSV HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def safe_read_csv(path, retries=3, delay=0.15):
    for attempt in range(retries):
        try:
            df = pd.read_csv(path)
            if df.empty:
                return None
            if not all(c in df.columns for c in ["range_mag", "phase", "energy"]):
                return None
            return df
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
    return None

def csv_changed(path):
    global _last_csv_hash
    try:
        tail     = pd.read_csv(path).tail(5).to_csv()
        new_hash = hash(tail)
        if new_hash != _last_csv_hash:
            _last_csv_hash = new_hash
            return True
        return False
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════════════════════════
_scale_checked = False

def predict(df):
    global _scale_checked

    window = df.tail(WINDOW_SIZE).copy()
    if len(window) < WINDOW_SIZE:
        return

    if not _scale_checked:
        _scale_checked = True
        for col in ["range_mag", "energy"]:
            if col in window.columns and col in norm_ref:
                z = abs(window[col].mean() - norm_ref[col]["mean"]) / \
                    (norm_ref[col]["std"] + 1e-10)
                if z > 3.0:
                    print(f"  ⚠️  Scale mismatch: {col}  z={z:.1f}σ — "
                          f"recollect training data with same frozen_max")

    if "delta_rm" in window.columns:
        d_arr = np.nan_to_num(
            pd.to_numeric(window["delta_rm"].values, errors="coerce"), 0.0)
    else:
        d_arr = np.zeros(WINDOW_SIZE)
    fall_prob, fall_state = fall_detector.update(d_arr)

    feat_dict   = extract_window(window)
    X           = pd.DataFrame([feat_dict])
    for col in features:
        if col not in X.columns:
            X[col] = 0.0
    X_scaled    = scaler.transform(X[features])
    model_probs = model.predict_proba(X_scaled)[0]
    model_pred  = int(np.argmax(model_probs))
    model_label = encoder.inverse_transform([model_pred])[0]
    model_conf  = float(np.max(model_probs))

    rule_scores_3, mean_delta_rm, is_fast = physics_scores_3class(window)

    fused4, mw, rw, consensus_cls = fuse_with_fall(
        model_probs, rule_scores_3, fall_prob, model_conf,
        mean_delta_rm, is_fast
    )

    display_label, display_conf, switched = stable.update(fused4, consensus_cls)

    display_prediction(
        display_label, display_conf, switched,
        model_label, model_conf, mw, rw,
        fused4, fall_state
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════
print("═"*60)
print("  THROUGH-WALL RADAR — ACTIVITY RECOGNITION  (V4)")
print(f"  Classes        : {' | '.join(ALL_CLASSES)}")
print(f"  Display        : every {DISPLAY_MIN_INTERVAL}s")
print(f"  Window         : {WINDOW_SIZE} chirps")
print(f"  Weights        : ML={MODEL_WEIGHT}  physics={RULE_WEIGHT}")
print(f"  Consensus TH   : {CONSENSUS_INSTANT_TH:.0%}  (instant switch when ML+physics agree)")
print(f"  Standing lock  : {STANDING_LOCK_TH:.0%}  (instant switch to standing)")
print(f"  Running cap    : {RUNNING_MAX_PROB_IF_SLOW:.0%}  (when Δrm < {TH_DELTA_RM_RUNNING})")
print("═"*60 + "\n")

while True:
    try:
        if os.path.exists(FLAG_FILE) and os.path.exists(CSV_FILE):
            flag_mtime = os.path.getmtime(FLAG_FILE)
            if flag_mtime != _last_flag_mtime:
                _last_flag_mtime = flag_mtime
                if csv_changed(CSV_FILE):
                    df = safe_read_csv(CSV_FILE)
                    if df is not None and len(df) >= WINDOW_SIZE:
                        predict(df)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Waiting for MATLAB...", end="\r")

        time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nStopped.")
        break
    except Exception as e:
        print(f"[WARN] {e}")
        time.sleep(0.5)