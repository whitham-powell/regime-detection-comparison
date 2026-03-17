# Regime Detection Comparison

Comparing two approaches to detecting regime changes in financial time series:
**Maximum Mean Discrepancy (MMD)** and **Bayesian Online Change Point Detection (BOCPD)**.

## Design

Both methods are applied to SPY daily data (2018–2024) in a 2×2 experimental
design crossing detection framework (MMD vs BOCPD) with feature dimensionality
(univariate log returns vs multivariate log OHLCV).

This repo owns no detection logic. It orchestrates three existing packages:

| Package | Role |
|---------|------|
| [finfeatures](https://github.com/whitham-powell/finfeatures) | OHLCV data fetching and feature engineering |
| [mmd-regime-detection](https://github.com/whitham-powell/mmd-regime-detection) | Nonparametric sliding-window MMD detection |
| [bocpd-regime-detection](https://github.com/whitham-powell/bocpd-regime-detection) | Bayesian online change point detection |

## Installation

```bash
# Using uv (recommended)
git clone https://github.com/whitham-powell/regime-detection-comparison.git
cd regime-detection-comparison
uv sync

# Or using pip
pip install -e ".[dev]"
```

### Google Colab

```python
!pip install git+https://github.com/whitham-powell/regime-detection-comparison.git
```

## Usage

The main notebook is `notebooks/comparison.py` (Jupytext percent format).

```bash
# Sync .py → .ipynb
make sync

# Execute and extract figures
make plots
```

## Results

Comparison axes:

- **Detection accuracy** against known market events (COVID crash, 2022 drawdown, etc.)
- **Sensitivity** to tuning parameters (window size, hazard rate)
- **Runtime** benchmarks
- **Output semantics** (soft probabilities vs hard boundaries)

See `PLAN.md` for the full experimental design.

## License

MIT
