# Implementation Plan: regime-detection-comparison

## Overview

A standalone comparison of two regime change detection approaches applied to
financial time series (SPY 2018–2024). This repo owns no detection logic — it
orchestrates three existing public packages as dependencies:

| Dependency | Role | Install |
|---|---|---|
| [finfeatures](https://github.com/whitham-powell/finfeatures) | Shared data layer — OHLCV fetching + feature engineering | `pip install git+https://...` |
| [mmd-regime-detection](https://github.com/whitham-powell/mmd-regime-detection) | Nonparametric MMD sliding-window detection | `pip install git+https://...` |
| [bocpd-regime-detection](https://github.com/whitham-powell/bocpd-regime-detection) | Bayesian Online Change Point Detection | `pip install git+https://...` |

The deliverable is a single notebook (`notebooks/comparison.py`) that runs a
2×2 experiment and produces presentation-ready figures and tables.

---

## Pre-work (required before this repo can run)

Two one-line changes in upstream repos to unblock Colab (Python 3.11):

1. **mmd-regime-detection/pyproject.toml**: change `requires-python = ">=3.13"`
   to `requires-python = ">=3.10"`
2. **bocpd-regime-detection/pyproject.toml**: same change

No code changes needed — only the version constraint.

---

## Experimental Design: 2×2 Matrix

|  | Univariate (log returns) | Multivariate (log OHLCV, 5 features) |
|---|---|---|
| **BOCPD** | `UnivariateNormalNIG` + `ConstantHazard` | `MultivariateNormalNIW` + `ConstantHazard` |
| **MMD** | RBF kernel on scalar signal | RBF kernel on 5-dim signal |

This separates two questions:
- **Framework effect** (compare within columns): nonparametric vs Bayesian, same features
- **Feature effect** (compare within rows): does multivariate input help each method?

### Feature Contract

| Signal | Source column(s) | Shape | Used by |
|--------|-----------------|-------|---------|
| `signal_uni` | `log_return` | (T,) or (T, 1) | MMD univariate, BOCPD univariate |
| `signal_multi` | `log_open, log_high, log_low, log_close, log_volume` | (T, 5) | MMD multivariate, BOCPD multivariate |

Both signals are standardized via `sklearn.preprocessing.StandardScaler`.
Both derived from `finfeatures` pipeline output.

### Known Events for Validation

```python
KNOWN_EVENTS = {
    "COVID crash":        ("2020-02-19", "2020-03-23"),
    "COVID recovery":     ("2020-03-24", "2020-08-01"),
    "2022 drawdown":      ("2022-01-03", "2022-10-12"),
    "2022 bottom":        ("2022-10-12", "2022-12-31"),
    "2023 acceleration":  ("2023-10-01", "2023-12-31"),
}
```

---

## Notebook Structure: `notebooks/comparison.py`

The notebook uses `# %%` cell markers (Jupytext percent format).
A `SAVE_FIGURES = False` flag at the top controls figure persistence.

### §1 — Data Pipeline (~3-4 cells)

**Goal**: Load SPY data, extract both signal variants, sanity-check plot.

```python
from finfeatures import minimal_pipeline
from finfeatures.sources import YFinanceSource

source = YFinanceSource()
raw = source.fetch("SPY", start="2018-01-01", end="2024-01-01")

# minimal_pipeline gives returns + log_returns
# We also need log OHLCV — check if minimal_pipeline includes these,
# otherwise use standard_pipeline or construct manually:
#   log_open  = np.log(raw["open"])
#   log_high  = np.log(raw["high"])
#   log_low   = np.log(raw["low"])
#   log_close = np.log(raw["close"])
#   log_volume = np.log(raw["volume"])
```

- Construct `signal_uni` and `signal_multi` as numpy arrays
- Standardize both with `StandardScaler` (fit on full series — this is
  exploratory, not predictive, so no train/test leakage concern)
- Define `KNOWN_EVENTS` dict
- Sanity plot: price + log returns, two-panel figure with date axis
- Print shape of both signals to confirm dimensions

### §2 — Detection Runs (~8 cells, 2 per config)

Each config gets a setup cell and a run cell. All timed with
`time.perf_counter()`. Results stored in a list of dicts with consistent keys.

**Common result structure** (dict per config):
```python
{
    "name": str,              # e.g. "BOCPD univariate"
    "method": str,            # "bocpd" or "mmd"
    "features": str,          # "univariate" or "multivariate"
    "boundaries": list,       # list of datetime boundary points
    "continuous_signal": pd.Series,  # z-score or P(cp), indexed by date
    "signal_label": str,      # "z-score from null" or "P(changepoint)"
    "runtime_seconds": float,
    "config": dict,           # hyperparameters used
}
```

#### 2a: BOCPD univariate
```python
from bocpd import BOCPD, ConstantHazard, UnivariateNormalNIG, extract_change_points_with_bounds

detector = BOCPD(
    model_factory=lambda: UnivariateNormalNIG(mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0),
    hazard_fn=ConstantHazard(lam=100),
)
result = detector.run(signal_uni)
boundaries = extract_change_points_with_bounds(result, credible_mass=0.90)
```

#### 2b: BOCPD multivariate
```python
from bocpd import MultivariateNormalNIW

detector = BOCPD(
    model_factory=lambda: MultivariateNormalNIW(...),  # check constructor args
    hazard_fn=ConstantHazard(lam=100),
)
result = detector.run(signal_multi)
boundaries = extract_change_points_with_bounds(result, credible_mass=0.90)
```

> **Note for implementer**: Check the `MultivariateNormalNIW` constructor
> signature. It likely takes prior parameters for the NIW distribution
> (mu0, kappa0, nu0, Psi0). The dimension will be 5 (log OHLCV).

#### 2c: MMD univariate
```python
from regime_detection import sliding_window_mmd, find_regime_boundaries, results_to_dataframe
from kta import rbf

sigma = np.median(np.abs(signal_uni - np.median(signal_uni)))
gamma = 1.0 / (2 * sigma**2)

results = sliding_window_mmd(
    data=signal_uni.reshape(-1, 1) if signal_uni.ndim == 1 else signal_uni,
    kernel_fn=rbf,
    kernel_params={"gamma": gamma},
    window=30,
    step=5,
    n_permutations=500,
)
results_df = results_to_dataframe(results, dates_index)
boundaries = find_regime_boundaries(results_df, threshold=10.0)
```

#### 2d: MMD multivariate
```python
sigma = np.median(np.abs(signal_multi - np.median(signal_multi)))
gamma = 1.0 / (2 * sigma**2)

results = sliding_window_mmd(
    data=signal_multi,
    kernel_fn=rbf,
    kernel_params={"gamma": gamma},
    window=30,
    step=5,
    n_permutations=500,
)
results_df = results_to_dataframe(results, dates_index)
boundaries = find_regime_boundaries(results_df, threshold=10.0)
```

### §3 — Results and Visualization (~6-8 cells)

#### Main panel figure (the hero figure)

4 vertically stacked subplots, shared x-axis (dates):

1. **Price panel**: SPY close price with vertical boundary lines from all
   four configs. Color coding:
   - Purple solid = MMD multivariate
   - Purple dashed = MMD univariate
   - Coral solid = BOCPD multivariate
   - Coral dashed = BOCPD univariate

2. **MMD signal panel**: z-score from null for both uni and multi runs,
   with horizontal threshold line at 10.0

3. **BOCPD signal panel**: P(changepoint) for both uni and multi runs,
   with horizontal threshold line (e.g. 0.5 or whatever is appropriate)

4. **Known events panel**: shaded regions from KNOWN_EVENTS dict with labels

Figure size: ~(14, 12). Save to `figures/main_panel.png` if SAVE_FIGURES.

#### Validation table

For each known event × each config:
- Detected: yes/no (any boundary within the event window)
- Offset: days from nearest boundary to event start date
- Signal strength: max continuous signal value within the event window

Render as a pandas DataFrame displayed in the notebook, and optionally
save as a figure via matplotlib table or as CSV.

#### Sensitivity sweeps

Two focused sweeps (not full factorial):

**MMD window sweep** on multivariate signal:
```python
window_sizes = [20, 30, 40, 60]
# For each: run sliding_window_mmd, extract boundaries, compute
# validation metrics against KNOWN_EVENTS
```

**BOCPD hazard sweep** on multivariate signal:
```python
hazard_lambdas = [50, 100, 200]
# For each: run BOCPD, extract boundaries, compute validation metrics
```

Present as small-multiple plots or a compact summary table showing
boundaries detected and mean offset from known events.

### §4 — Comparison and Discussion (~4-5 markdown + figure cells)

#### Normalized signal overlay
Both continuous signals (MMD z-score and BOCPD P(cp)) normalized to [0, 1]
on the same plot. Shows where the methods agree and disagree about regime
change timing.

#### Runtime table
2×2 table of wall-clock times from §2. Note scaling properties:
- BOCPD: online, O(T²) naive
- MMD: batch, O(n_windows × n_perms × window²)

#### Assumptions comparison table

| Aspect | BOCPD | MMD |
|--------|-------|-----|
| **Distributional assumption** | Gaussian (NIG/NIW conjugate) | None (characteristic kernel) |
| **Online capable** | Yes (by design) | No (requires full windows) |
| **Output type** | P(changepoint) ∈ [0,1] with credible intervals | z-score from permutation null |
| **Key tuning parameters** | Hazard rate λ, prior strength (κ₀, α₀, β₀ / ν₀, Ψ₀) | Window size, kernel bandwidth, significance threshold |
| **Multivariate** | Via NIW (models covariance) | Native (kernel on feature vectors) |
| **Computational cost** | Fast (closed-form updates) | Expensive (permutation test per window) |

#### Discussion markdown cells

- **Feature effect (rows)**: Does 5-dim log OHLCV improve detection for each
  method compared to univariate log returns? Where do the improvements show
  up — major events or subtler transitions?

- **Framework effect (columns)**: Holding features constant, what does each
  method catch that the other misses? BOCPD may react faster (online), while
  MMD may be more robust to non-Gaussian shifts.

- **Practical takeaways**: When would you choose one over the other?
  BOCPD for real-time monitoring with calibrated uncertainty. MMD for
  offline analysis where you want sensitivity to arbitrary distributional
  changes.

---

## Repo Configuration Details

### pyproject.toml
- `requires-python = ">=3.10"`
- Dependencies: finfeatures[yfinance], regime-detection, bocpd (all from git)
  plus matplotlib, scikit-learn, numpy, pandas
- Dev dependencies: jupytext, pre-commit, black, isort

### Makefile targets
- `sync`: jupytext --sync on notebooks/
- `plots`: execute notebooks, extract figures to figures/
- `clean`: remove generated .ipynb and figures

### .pre-commit-config.yaml
- jupytext sync for notebooks/**/*.py
- black + isort on all .py files

### Conventions
- `SAVE_FIGURES = False` at notebook top (toggled by make plots or manually)
- Figures saved to `figures/` with descriptive names
- No code duplication from dependency repos — import only
- `.py` with `# %%` is source of truth; `.ipynb` is generated artifact
