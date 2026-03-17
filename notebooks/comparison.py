# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Regime Detection Comparison: MMD vs BOCPD
#
# Comparing nonparametric (MMD) and Bayesian (BOCPD) approaches to detecting
# regime changes in SPY daily returns (2018–2024).
#
# **2×2 design**: each method is run on both univariate (log returns) and
# multivariate (log OHLCV) signals to separate the framework effect from the
# feature effect.

# %% [markdown]
# ## Configuration

# %%
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SAVE_FIGURES = os.environ.get("SAVE_FIGURES", "0") == "1"
FIGURES_DIR = "../figures"

KNOWN_EVENTS = {
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "COVID recovery": ("2020-03-24", "2020-08-01"),
    "2022 drawdown": ("2022-01-03", "2022-10-12"),
    "2022 bottom": ("2022-10-12", "2022-12-31"),
    "2023 acceleration": ("2023-10-01", "2023-12-31"),
}


def save_fig(fig, name):
    if SAVE_FIGURES:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(
            os.path.join(FIGURES_DIR, name), dpi=200, bbox_inches="tight"
        )


# %% [markdown]
# ---
# ## §1 — Data Pipeline
#
# Fetch SPY via finfeatures, extract univariate and multivariate signal
# variants, standardize.

# %%
from finfeatures.sources import YFinanceSource

source = YFinanceSource()
raw = source.fetch("SPY", start="2018-01-01", end="2024-01-01")
print(f"Raw data: {raw.shape[0]} rows, columns: {list(raw.columns)}")
raw.head()

# %%
# Construct signals from raw OHLCV
# Univariate: log returns
log_returns = np.log(raw["close"] / raw["close"].shift(1)).dropna()

# Multivariate: log OHLCV (5 features)
log_ohlcv = pd.DataFrame(
    {
        "log_open": np.log(raw["open"]),
        "log_high": np.log(raw["high"]),
        "log_low": np.log(raw["low"]),
        "log_close": np.log(raw["close"]),
        "log_volume": np.log(raw["volume"]),
    },
    index=raw.index,
).dropna()

# Align indices (log_returns has one fewer row due to differencing)
common_idx = log_returns.index.intersection(log_ohlcv.index)
log_returns = log_returns.loc[common_idx]
log_ohlcv = log_ohlcv.loc[common_idx]

# Standardize
scaler_uni = StandardScaler()
signal_uni = scaler_uni.fit_transform(log_returns.values.reshape(-1, 1)).ravel()

scaler_multi = StandardScaler()
signal_multi = scaler_multi.fit_transform(log_ohlcv.values)

dates_index = common_idx

print(f"Univariate signal:   shape {signal_uni.shape}")
print(f"Multivariate signal: shape {signal_multi.shape}")
print(f"Date range: {dates_index[0].date()} to {dates_index[-1].date()}")

# %%
# Sanity check: price and log returns
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

axes[0].plot(raw.index, raw["close"], color="black", linewidth=0.8)
axes[0].set_ylabel("SPY Close")
axes[0].set_title("SPY Price and Log Returns (2018–2024)")

axes[1].plot(dates_index, log_returns.values, color="steelblue", linewidth=0.4)
axes[1].set_ylabel("Log Return")
axes[1].set_xlabel("Date")

# Shade known events
for label, (start, end) in KNOWN_EVENTS.items():
    for ax in axes:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.1, color="gray")

plt.tight_layout()
save_fig(fig, "sanity_check.png")
plt.show()


# %% [markdown]
# ---
# ## §2 — Detection Runs
#
# Four configurations in the 2×2 design. Each is timed and produces a
# consistent result dict.

# %%
# Container for all results
all_results = []

# %% [markdown]
# ### 2a: BOCPD — Univariate (log returns)

# %%
from bocpd import (
    BOCPD,
    ConstantHazard,
    UnivariateNormalNIG,
    extract_change_points_with_bounds,
)

t0 = time.perf_counter()

detector_bocpd_uni = BOCPD(
    model_factory=lambda: UnivariateNormalNIG(
        mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0
    ),
    hazard_fn=ConstantHazard(lam=100),
)
bocpd_uni_raw = detector_bocpd_uni.run(signal_uni)
bocpd_uni_boundaries = extract_change_points_with_bounds(
    bocpd_uni_raw, credible_mass=0.90
)

runtime_bocpd_uni = time.perf_counter() - t0

# TODO: Extract continuous P(changepoint) signal from bocpd_uni_raw
# This depends on the BOCPD result object structure — likely
# bocpd_uni_raw.changepoint_probability or similar.
# bocpd_uni_cp_prob = ...

all_results.append(
    {
        "name": "BOCPD univariate",
        "method": "bocpd",
        "features": "univariate",
        "boundaries": [
            dates_index[b["index"]] for b in bocpd_uni_boundaries
        ],
        # "continuous_signal": pd.Series(bocpd_uni_cp_prob, index=dates_index),
        "signal_label": "P(changepoint)",
        "runtime_seconds": runtime_bocpd_uni,
        "config": {"hazard_lambda": 100, "kappa0": 0.1, "alpha0": 1.0, "beta0": 1.0},
    }
)

print(f"BOCPD univariate: {len(bocpd_uni_boundaries)} boundaries, {runtime_bocpd_uni:.2f}s")
for b in bocpd_uni_boundaries:
    print(f"  t={b['index']} ({dates_index[b['index']].date()}), "
          f"90% CI: [{b['lower']}, {b['upper']}]")

# %% [markdown]
# ### 2b: BOCPD — Multivariate (log OHLCV)

# %%
from bocpd import MultivariateNormalNIW

t0 = time.perf_counter()

# TODO: Verify MultivariateNormalNIW constructor signature.
# Likely takes: mu0, kappa0, nu0, Psi0 where Psi0 is a (D, D) scale matrix.
D = signal_multi.shape[1]  # 5

detector_bocpd_multi = BOCPD(
    model_factory=lambda: MultivariateNormalNIW(
        mu0=np.zeros(D),
        kappa0=0.1,
        nu0=D + 2,          # must be > D - 1 for valid Wishart
        Psi0=np.eye(D),
    ),
    hazard_fn=ConstantHazard(lam=100),
)
bocpd_multi_raw = detector_bocpd_multi.run(signal_multi)
bocpd_multi_boundaries = extract_change_points_with_bounds(
    bocpd_multi_raw, credible_mass=0.90
)

runtime_bocpd_multi = time.perf_counter() - t0

all_results.append(
    {
        "name": "BOCPD multivariate",
        "method": "bocpd",
        "features": "multivariate",
        "boundaries": [
            dates_index[b["index"]] for b in bocpd_multi_boundaries
        ],
        "signal_label": "P(changepoint)",
        "runtime_seconds": runtime_bocpd_multi,
        "config": {"hazard_lambda": 100, "kappa0": 0.1, "nu0": D + 2},
    }
)

print(f"BOCPD multivariate: {len(bocpd_multi_boundaries)} boundaries, {runtime_bocpd_multi:.2f}s")
for b in bocpd_multi_boundaries:
    print(f"  t={b['index']} ({dates_index[b['index']].date()}), "
          f"90% CI: [{b['lower']}, {b['upper']}]")

# %% [markdown]
# ### 2c: MMD — Univariate (log returns)

# %%
from kta import rbf
from regime_detection import (
    find_regime_boundaries,
    results_to_dataframe,
    sliding_window_mmd,
)

t0 = time.perf_counter()

signal_uni_2d = signal_uni.reshape(-1, 1)
sigma_uni = np.median(np.abs(signal_uni_2d - np.median(signal_uni_2d)))
gamma_uni = 1.0 / (2 * sigma_uni**2)

mmd_uni_raw = sliding_window_mmd(
    data=signal_uni_2d,
    kernel_fn=rbf,
    kernel_params={"gamma": gamma_uni},
    window=30,
    step=5,
    n_permutations=500,
)
mmd_uni_df = results_to_dataframe(mmd_uni_raw, dates_index)
mmd_uni_boundaries = find_regime_boundaries(mmd_uni_df, threshold=10.0)

runtime_mmd_uni = time.perf_counter() - t0

all_results.append(
    {
        "name": "MMD univariate",
        "method": "mmd",
        "features": "univariate",
        "boundaries": mmd_uni_boundaries,  # TODO: verify this is list of dates
        "continuous_signal": mmd_uni_df["std_from_null"] if "std_from_null" in mmd_uni_df.columns else None,
        "signal_label": "z-score from null",
        "runtime_seconds": runtime_mmd_uni,
        "config": {"window": 30, "step": 5, "n_permutations": 500, "gamma": gamma_uni, "threshold": 10.0},
    }
)

print(f"MMD univariate: {len(mmd_uni_boundaries)} boundaries, {runtime_mmd_uni:.2f}s")

# %% [markdown]
# ### 2d: MMD — Multivariate (log OHLCV)

# %%
t0 = time.perf_counter()

sigma_multi = np.median(np.abs(signal_multi - np.median(signal_multi)))
gamma_multi = 1.0 / (2 * sigma_multi**2)

mmd_multi_raw = sliding_window_mmd(
    data=signal_multi,
    kernel_fn=rbf,
    kernel_params={"gamma": gamma_multi},
    window=30,
    step=5,
    n_permutations=500,
)
mmd_multi_df = results_to_dataframe(mmd_multi_raw, dates_index)
mmd_multi_boundaries = find_regime_boundaries(mmd_multi_df, threshold=10.0)

runtime_mmd_multi = time.perf_counter() - t0

all_results.append(
    {
        "name": "MMD multivariate",
        "method": "mmd",
        "features": "multivariate",
        "boundaries": mmd_multi_boundaries,
        "continuous_signal": mmd_multi_df["std_from_null"] if "std_from_null" in mmd_multi_df.columns else None,
        "signal_label": "z-score from null",
        "runtime_seconds": runtime_mmd_multi,
        "config": {"window": 30, "step": 5, "n_permutations": 500, "gamma": gamma_multi, "threshold": 10.0},
    }
)

print(f"MMD multivariate: {len(mmd_multi_boundaries)} boundaries, {runtime_mmd_multi:.2f}s")

# %% [markdown]
# ---
# ## §3 — Results and Visualization

# %% [markdown]
# ### Main panel figure
#
# Four vertically stacked panels with shared date axis:
# 1. Price + boundary lines (all four configs)
# 2. MMD z-scores (uni + multi)
# 3. BOCPD P(changepoint) (uni + multi)
# 4. Known events as shaded regions

# %%
# TODO: Build the main 4-panel figure.
#
# Color scheme:
#   MMD:   purple (#534AB7 solid multi, dashed uni)
#   BOCPD: coral  (#D85A30 solid multi, dashed uni)
#
# Boundary lines: axvline with appropriate color/linestyle per config.
#
# Panel 2 (MMD signals): plot mmd_uni_df and mmd_multi_df z-score columns
# Panel 3 (BOCPD signals): plot changepoint probability from both runs
# Panel 4 (events): axvspan shaded regions with text labels
#
# This is the hero figure — spend time making it clean.
# fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
# ...
# save_fig(fig, "main_panel.png")

pass

# %% [markdown]
# ### Validation table
#
# For each known event × each config: detected (yes/no), days offset from
# event start, signal strength at peak within the event window.

# %%
# TODO: Build validation table.
#
# For each result in all_results:
#   For each event in KNOWN_EVENTS:
#     - Check if any boundary falls within (event_start, event_end)
#     - Compute min days offset from any boundary to event_start
#     - Find peak continuous signal value in the event window
#
# Display as a pandas DataFrame with multi-level columns:
#   (config_name, detected), (config_name, offset_days), ...
#
# validation_records = []
# ...
# validation_df = pd.DataFrame(validation_records)
# validation_df

pass

# %% [markdown]
# ### Sensitivity sweeps

# %% [markdown]
# #### MMD window size sweep (multivariate signal)

# %%
# TODO: Sweep window sizes [20, 30, 40, 60].
# For each: run sliding_window_mmd on signal_multi, extract boundaries,
# compute number of boundaries and mean offset from known events.
#
# window_sizes = [20, 30, 40, 60]
# mmd_sweep_results = []
# for w in window_sizes:
#     ...

pass

# %% [markdown]
# #### BOCPD hazard rate sweep (multivariate signal)

# %%
# TODO: Sweep hazard lambdas [50, 100, 200].
# For each: run BOCPD with MultivariateNormalNIW, extract boundaries,
# compute number of boundaries and mean offset from known events.
#
# hazard_lambdas = [50, 100, 200]
# bocpd_sweep_results = []
# for lam in hazard_lambdas:
#     ...

pass


# %% [markdown]
# ---
# ## §4 — Comparison and Discussion

# %% [markdown]
# ### Normalized signal overlay
#
# Both continuous signals (MMD z-score and BOCPD P(changepoint)) min-max
# normalized to [0, 1] on the same plot. Shows where the methods agree
# and disagree about regime change timing.

# %%
# TODO: Normalize both continuous signals to [0, 1] and overlay.
# Use multivariate configs as the primary comparison.
#
# fig, ax = plt.subplots(figsize=(14, 4))
# ...
# save_fig(fig, "normalized_overlay.png")

pass

# %% [markdown]
# ### Runtime comparison

# %%
runtime_df = pd.DataFrame(
    [
        {"Method": r["name"], "Runtime (s)": f"{r['runtime_seconds']:.2f}"}
        for r in all_results
    ]
)
print(runtime_df.to_string(index=False))

# %% [markdown]
# ### Assumptions comparison
#
# | Aspect | BOCPD | MMD |
# |--------|-------|-----|
# | **Distributional assumption** | Gaussian (NIG univariate / NIW multivariate) | None (characteristic kernel) |
# | **Online capable** | Yes (by design) | No (requires full windows) |
# | **Output type** | P(changepoint) ∈ [0,1] with credible intervals | z-score from permutation null |
# | **Key tuning parameters** | Hazard rate λ, prior hyperparameters | Window size, kernel bandwidth, threshold |
# | **Multivariate** | Via NIW (models full covariance) | Native (kernel on feature vectors) |
# | **Computational cost** | Fast (closed-form conjugate updates) | Expensive (permutation test per window) |

# %% [markdown]
# ### Discussion
#
# **Feature effect (comparing rows):**
#
# TODO: Does going from univariate log returns to 5-dim log OHLCV improve
# detection for each method? Where do improvements appear — major events
# that both catch, or subtler transitions?
#
# **Framework effect (comparing columns):**
#
# TODO: Holding features constant, what does each method catch that the other
# misses? BOCPD reacts online; MMD requires full windows but makes no
# distributional assumptions.
#
# **Practical takeaways:**
#
# TODO: When would you choose one over the other?
# - BOCPD: real-time monitoring, calibrated uncertainty, Gaussian-regime assumption is acceptable
# - MMD: offline analysis, sensitivity to arbitrary distributional changes, multivariate-native
