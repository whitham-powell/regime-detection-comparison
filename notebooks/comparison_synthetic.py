# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Synthetic Experiments: MMD vs BOCPD
#
# Controlled experiments with planted change points to measure precision,
# recall, detection latency, and credible-interval coverage — something
# impossible with the real SPY data where ground-truth boundaries are unknown.
#
# **Experiment A** — Gaussian synthetic data (mean shift + variance shift):
# baseline accuracy when BOCPD's model is correctly specified.
#
# **Experiment B** — Student-t synthetic data (df = 30 / 5 / 3): tests how
# BOCPD's Gaussian assumption degrades under heavy tails while MMD remains
# nonparametric.

# %% [markdown]
# ## S0 — Configuration

# %%
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from bocpd import (
    BOCPD,
    ConstantHazard,
    UnivariateNormalNIG,
    extract_change_points_with_bounds,
)
from kta import rbf
from regime_detection import (
    find_regime_boundaries,
    results_to_dataframe,
    sliding_window_mmd,
)

SAVE_FIGURES = os.environ.get("SAVE_FIGURES", "0") == "1"
FIGURES_DIR = "../figures"

SEGMENT_LENGTH = 200
N_SEGMENTS = 5
TOTAL_LENGTH = SEGMENT_LENGTH * N_SEGMENTS  # 1000

BOCPD_R_MAX = 600
BOCPD_HAZARD_LAMBDA = 100
MMD_WINDOW = 30
MMD_STEP = 5
MMD_PERMUTATIONS = 500
MMD_THRESHOLD = 10.0

DETECTION_TOLERANCE = 20  # index positions

rng = np.random.default_rng(42)


def save_fig(fig, name):
    if SAVE_FIGURES:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(
            os.path.join(FIGURES_DIR, name), dpi=200, bbox_inches="tight"
        )


# %% [markdown]
# ### Helper functions

# %%
def make_synthetic_signal(segment_lengths, means, stds, rng=rng):
    """Generate piecewise-Gaussian signal with known change points."""
    segments = []
    for length, mu, sigma in zip(segment_lengths, means, stds):
        segments.append(rng.normal(loc=mu, scale=sigma, size=length))
    X = np.concatenate(segments)
    true_cps = np.cumsum(segment_lengths)[:-1]  # boundaries between segments
    return X, true_cps


def make_heavy_tailed_signal(segment_lengths, means, scales, dfs, rng=rng):
    """Generate piecewise Student-t signal with known change points."""
    segments = []
    for length, mu, scale, df in zip(segment_lengths, means, scales, dfs):
        samples = rng.standard_t(df, size=length) * scale + mu
        segments.append(samples)
    X = np.concatenate(segments)
    true_cps = np.cumsum(segment_lengths)[:-1]
    return X, true_cps


def score_detections(true_cps, detected, tolerance=DETECTION_TOLERANCE):
    """Score detected change points against ground truth (greedy nearest-first).

    Returns dict with tp, fp, fn, precision, recall, mean_latency, latencies.
    """
    true_set = list(true_cps)
    detected = sorted(detected)
    matched_true = set()
    matched_det = set()
    latencies = []

    # Build distance pairs sorted by distance, then greedily match
    pairs = []
    for i, t in enumerate(true_set):
        for j, d in enumerate(detected):
            dist = abs(int(d) - int(t))
            if dist <= tolerance:
                pairs.append((dist, i, j))
    pairs.sort()

    for dist, i, j in pairs:
        if i not in matched_true and j not in matched_det:
            matched_true.add(i)
            matched_det.add(j)
            latencies.append(int(detected[j]) - int(true_set[i]))

    tp = len(matched_true)
    fp = len(detected) - tp
    fn = len(true_set) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "mean_latency": np.mean(latencies) if latencies else float("nan"),
        "latencies": latencies,
    }


def ci_coverage(true_cps, boundaries_with_bounds, credible_mass=0.90):
    """Fraction of true CPs falling within BOCPD credible intervals."""
    if len(boundaries_with_bounds) == 0 or len(true_cps) == 0:
        return float("nan")
    covered = 0
    for tcp in true_cps:
        for b in boundaries_with_bounds:
            if b["lower"] <= tcp <= b["upper"]:
                covered += 1
                break
    return covered / len(true_cps)


def run_bocpd_on_synthetic(X):
    """Run BOCPD on a 1-D synthetic signal.

    Returns (boundaries_int, raw_result, boundaries_with_bounds).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, 1)).ravel()

    detector = BOCPD(
        model_factory=lambda: UnivariateNormalNIG(
            mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0
        ),
        hazard_fn=ConstantHazard(lam=BOCPD_HAZARD_LAMBDA),
        r_max=BOCPD_R_MAX,
    )
    result = detector.run(X_scaled)
    bwb = extract_change_points_with_bounds(result, credible_mass=0.90)
    boundaries = [b["index"] for b in bwb]
    return boundaries, result, bwb


def run_mmd_on_synthetic(X):
    """Run MMD on a 1-D synthetic signal.

    Returns (boundaries_int, mmd_df).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, 1))  # keep 2-D for MMD

    sigma = np.median(np.abs(X_scaled - np.median(X_scaled)))
    gamma = 1.0 / (2 * sigma**2)

    # MMD API requires DatetimeIndex — create dummy daily dates
    dates = pd.date_range("2024-01-01", periods=len(X), freq="D")

    raw = sliding_window_mmd(
        data=X_scaled,
        kernel_fn=rbf,
        kernel_params={"gamma": gamma},
        window=MMD_WINDOW,
        step=MMD_STEP,
        n_permutations=MMD_PERMUTATIONS,
    )
    mmd_df = results_to_dataframe(raw, dates)
    mmd_boundary_dates = find_regime_boundaries(
        mmd_df, threshold=MMD_THRESHOLD, min_gap_days=20
    )
    # Convert dates back to integer positions
    boundaries = [dates.get_loc(d) for d in mmd_boundary_dates]
    return boundaries, mmd_df


# %% [markdown]
# ### Plotting helpers

# %%
COLOR_BOCPD = "#D85A30"
COLOR_MMD = "#534AB7"
COLOR_TRUE = "black"


def plot_three_panel(
    X, true_cps, bocpd_bounds, mmd_bounds, bocpd_result, mmd_df, title
):
    """3-panel figure: signal + boundaries, BOCPD E[RL], MMD z-score."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    idx = np.arange(len(X))

    # Panel 1 — signal with detected and true change points
    axes[0].plot(idx, X, color="gray", linewidth=0.5, alpha=0.7)
    for cp in true_cps:
        axes[0].axvline(cp, color=COLOR_TRUE, linestyle="--", linewidth=1.2, alpha=0.8)
    for cp in bocpd_bounds:
        axes[0].axvline(cp, color=COLOR_BOCPD, linestyle="-", linewidth=1.0, alpha=0.7)
    for cp in mmd_bounds:
        axes[0].axvline(cp, color=COLOR_MMD, linestyle="-", linewidth=1.0, alpha=0.7)
    axes[0].set_ylabel("Signal")
    axes[0].set_title(title)
    # Legend
    axes[0].plot([], [], color=COLOR_TRUE, linestyle="--", label="True CP")
    axes[0].plot([], [], color=COLOR_BOCPD, linestyle="-", label="BOCPD")
    axes[0].plot([], [], color=COLOR_MMD, linestyle="-", label="MMD")
    axes[0].legend(loc="upper right", fontsize=8)

    # Panel 2 — BOCPD expected run length
    erl = bocpd_result["expected_run_length"]
    axes[1].plot(idx, erl, color=COLOR_BOCPD, linewidth=0.8)
    for cp in true_cps:
        axes[1].axvline(cp, color=COLOR_TRUE, linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("E[run length]")

    # Panel 3 — MMD z-score
    mmd_t = mmd_df["t"].values.astype(int)
    mmd_z = mmd_df["std_from_null"].values
    axes[2].plot(mmd_t, mmd_z, color=COLOR_MMD, linewidth=0.8)
    axes[2].axhline(MMD_THRESHOLD, color="gray", linestyle=":", linewidth=0.8)
    for cp in true_cps:
        axes[2].axvline(cp, color=COLOR_TRUE, linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].set_ylabel("MMD z-score")
    axes[2].set_xlabel("Observation index")

    plt.tight_layout()
    return fig


# %% [markdown]
# ---
# ## S1 — Experiment A: Gaussian Synthetic Data
#
# BOCPD assumes Gaussian observations, so this is the correctly-specified
# setting. Two sub-experiments test mean shifts and variance shifts separately.

# %% [markdown]
# ### A1: Mean Shift

# %%
seg_lengths_A = [SEGMENT_LENGTH] * N_SEGMENTS
means_A1 = [0, 2, 0, -1, 1]
stds_A1 = [1.0] * N_SEGMENTS

X_A1, true_cps_A1 = make_synthetic_signal(seg_lengths_A, means_A1, stds_A1)
print(f"A1 signal: {len(X_A1)} observations, true CPs at {list(true_cps_A1)}")

# %%
t0 = time.perf_counter()
bocpd_A1, bocpd_A1_raw, bocpd_A1_bwb = run_bocpd_on_synthetic(X_A1)
bocpd_A1_time = time.perf_counter() - t0

t0 = time.perf_counter()
mmd_A1, mmd_A1_df = run_mmd_on_synthetic(X_A1)
mmd_A1_time = time.perf_counter() - t0

print(f"BOCPD detected: {bocpd_A1}  ({bocpd_A1_time:.1f}s)")
print(f"MMD   detected: {mmd_A1}  ({mmd_A1_time:.1f}s)")

# %%
fig_A1 = plot_three_panel(
    X_A1, true_cps_A1, bocpd_A1, mmd_A1,
    bocpd_A1_raw, mmd_A1_df,
    "Experiment A1: Mean Shift (Gaussian)",
)
save_fig(fig_A1, "synthetic_A1_mean_shift.png")
plt.show()

# %% [markdown]
# ### A2: Variance Shift

# %%
means_A2 = [0.0] * N_SEGMENTS
stds_A2 = [0.5, 2.0, 0.5, 1.5, 0.5]

X_A2, true_cps_A2 = make_synthetic_signal(seg_lengths_A, means_A2, stds_A2)
print(f"A2 signal: {len(X_A2)} observations, true CPs at {list(true_cps_A2)}")

# %%
t0 = time.perf_counter()
bocpd_A2, bocpd_A2_raw, bocpd_A2_bwb = run_bocpd_on_synthetic(X_A2)
bocpd_A2_time = time.perf_counter() - t0

t0 = time.perf_counter()
mmd_A2, mmd_A2_df = run_mmd_on_synthetic(X_A2)
mmd_A2_time = time.perf_counter() - t0

print(f"BOCPD detected: {bocpd_A2}  ({bocpd_A2_time:.1f}s)")
print(f"MMD   detected: {mmd_A2}  ({mmd_A2_time:.1f}s)")

# %%
fig_A2 = plot_three_panel(
    X_A2, true_cps_A2, bocpd_A2, mmd_A2,
    bocpd_A2_raw, mmd_A2_df,
    "Experiment A2: Variance Shift (Gaussian)",
)
save_fig(fig_A2, "synthetic_A2_variance_shift.png")
plt.show()

# %% [markdown]
# ### Experiment A — Metrics Summary

# %%
def build_metrics_row(label, true_cps, bocpd_bounds, mmd_bounds, bocpd_bwb):
    """Build a metrics dict for one experiment."""
    bocpd_scores = score_detections(true_cps, bocpd_bounds)
    mmd_scores = score_detections(true_cps, mmd_bounds)
    bocpd_cov = ci_coverage(true_cps, bocpd_bwb)
    return [
        {
            "experiment": label,
            "method": "BOCPD",
            "precision": bocpd_scores["precision"],
            "recall": bocpd_scores["recall"],
            "mean_latency": bocpd_scores["mean_latency"],
            "fp": bocpd_scores["fp"],
            "ci_coverage": bocpd_cov,
        },
        {
            "experiment": label,
            "method": "MMD",
            "precision": mmd_scores["precision"],
            "recall": mmd_scores["recall"],
            "mean_latency": mmd_scores["mean_latency"],
            "fp": mmd_scores["fp"],
            "ci_coverage": float("nan"),  # MMD has no CI
        },
    ]


rows = []
rows += build_metrics_row("A1: Mean shift", true_cps_A1, bocpd_A1, mmd_A1, bocpd_A1_bwb)
rows += build_metrics_row("A2: Variance shift", true_cps_A2, bocpd_A2, mmd_A2, bocpd_A2_bwb)

metrics_A = pd.DataFrame(rows)
print("=== Experiment A: Metrics ===")
print(metrics_A.to_string(index=False, float_format="%.3f"))

# %% [markdown]
# **Discussion — Experiment A**
#
# Both methods should detect mean shifts well (A1). BOCPD's NIG model
# explicitly tracks variance, so it may have an edge on pure variance shifts
# (A2), where the RBF kernel bandwidth becomes the limiting factor for MMD.
# CI coverage tells us whether the 90% credible intervals from the
# run-length posterior are well-calibrated under correct specification.

# %% [markdown]
# ---
# ## S2 — Experiment B: Gaussian Assumption Violation
#
# We repeat the mean-shift task from A1 but generate observations from
# Student-t distributions with decreasing degrees of freedom. As df drops,
# the tails grow heavier and BOCPD's Gaussian assumption becomes
# increasingly misspecified.

# %% [markdown]
# ### B1: df = 30 (near-Gaussian control)

# %%
dfs_B1 = [30] * N_SEGMENTS
means_B = means_A1  # same mean-shift pattern as A1
scales_B = [1.0] * N_SEGMENTS

X_B1, true_cps_B = make_heavy_tailed_signal(seg_lengths_A, means_B, scales_B, dfs_B1)
print(f"B1 signal: {len(X_B1)} observations, df=30")

t0 = time.perf_counter()
bocpd_B1, bocpd_B1_raw, bocpd_B1_bwb = run_bocpd_on_synthetic(X_B1)
bocpd_B1_time = time.perf_counter() - t0

t0 = time.perf_counter()
mmd_B1, mmd_B1_df = run_mmd_on_synthetic(X_B1)
mmd_B1_time = time.perf_counter() - t0

print(f"BOCPD detected: {bocpd_B1}  ({bocpd_B1_time:.1f}s)")
print(f"MMD   detected: {mmd_B1}  ({mmd_B1_time:.1f}s)")

# %% [markdown]
# ### B2: df = 5 (moderate tails)

# %%
dfs_B2 = [5] * N_SEGMENTS

X_B2, _ = make_heavy_tailed_signal(seg_lengths_A, means_B, scales_B, dfs_B2)
print(f"B2 signal: {len(X_B2)} observations, df=5")

t0 = time.perf_counter()
bocpd_B2, bocpd_B2_raw, bocpd_B2_bwb = run_bocpd_on_synthetic(X_B2)
bocpd_B2_time = time.perf_counter() - t0

t0 = time.perf_counter()
mmd_B2, mmd_B2_df = run_mmd_on_synthetic(X_B2)
mmd_B2_time = time.perf_counter() - t0

print(f"BOCPD detected: {bocpd_B2}  ({bocpd_B2_time:.1f}s)")
print(f"MMD   detected: {mmd_B2}  ({mmd_B2_time:.1f}s)")

# %% [markdown]
# ### B3: df = 3 (heavy tails)

# %%
dfs_B3 = [3] * N_SEGMENTS

X_B3, _ = make_heavy_tailed_signal(seg_lengths_A, means_B, scales_B, dfs_B3)
print(f"B3 signal: {len(X_B3)} observations, df=3")

t0 = time.perf_counter()
bocpd_B3, bocpd_B3_raw, bocpd_B3_bwb = run_bocpd_on_synthetic(X_B3)
bocpd_B3_time = time.perf_counter() - t0

t0 = time.perf_counter()
mmd_B3, mmd_B3_df = run_mmd_on_synthetic(X_B3)
mmd_B3_time = time.perf_counter() - t0

print(f"BOCPD detected: {bocpd_B3}  ({bocpd_B3_time:.1f}s)")
print(f"MMD   detected: {mmd_B3}  ({mmd_B3_time:.1f}s)")

# %%
fig_B3 = plot_three_panel(
    X_B3, true_cps_B, bocpd_B3, mmd_B3,
    bocpd_B3_raw, mmd_B3_df,
    "Experiment B3: Mean Shift with Heavy Tails (df=3)",
)
save_fig(fig_B3, "synthetic_B3_heavy_tails.png")
plt.show()

# %% [markdown]
# ### Experiment B — Metrics Summary

# %%
rows_B = []
rows_B += build_metrics_row("B1: df=30", true_cps_B, bocpd_B1, mmd_B1, bocpd_B1_bwb)
rows_B += build_metrics_row("B2: df=5", true_cps_B, bocpd_B2, mmd_B2, bocpd_B2_bwb)
rows_B += build_metrics_row("B3: df=3", true_cps_B, bocpd_B3, mmd_B3, bocpd_B3_bwb)

metrics_B = pd.DataFrame(rows_B)
print("=== Experiment B: Metrics ===")
print(metrics_B.to_string(index=False, float_format="%.3f"))

# %% [markdown]
# ### Degradation Summary

# %%
fig_deg, axes_deg = plt.subplots(1, 2, figsize=(12, 5))

# Extract BOCPD and MMD metrics for B1-B3
for method, color in [("BOCPD", COLOR_BOCPD), ("MMD", COLOR_MMD)]:
    sub = metrics_B[metrics_B["method"] == method]
    labels = ["df=30", "df=5", "df=3"]
    x = np.arange(len(labels))
    width = 0.35
    offset = -width / 2 if method == "BOCPD" else width / 2

    axes_deg[0].bar(
        x + offset, sub["precision"].values, width,
        label=method, color=color, alpha=0.8,
    )
    axes_deg[1].bar(
        x + offset, sub["recall"].values, width,
        label=method, color=color, alpha=0.8,
    )

for ax, metric_name in zip(axes_deg, ["Precision", "Recall"]):
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["df=30", "df=5", "df=3"])
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_xlabel("Student-t degrees of freedom")

axes_deg[0].set_title("Precision Degradation under Heavy Tails")
axes_deg[1].set_title("Recall Degradation under Heavy Tails")

plt.tight_layout()
save_fig(fig_deg, "synthetic_B_degradation.png")
plt.show()

# %% [markdown]
# **Discussion — Experiment B**
#
# - **B1 (df=30)**: Near-Gaussian; results should closely match A1.
# - **B2 (df=5)**: Moderate tails introduce occasional large observations
#   that BOCPD may mistake for regime changes, increasing false positives
#   and degrading precision.
# - **B3 (df=3)**: Heavy tails cause frequent outliers. BOCPD's credible
#   intervals, calibrated under Gaussianity, lose coverage. MMD's kernel
#   saturates on outliers, making it relatively more stable.
#
# The degradation chart above quantifies how the Gaussian assumption
# breaks down, motivating the use of heavier-tailed observation models
# (e.g., Student-t likelihood) or the nonparametric MMD approach when
# tail risk is present.
