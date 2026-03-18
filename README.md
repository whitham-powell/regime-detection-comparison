# Regime Detection Comparison

Comparing two approaches to detecting regime changes in financial time series:
**Maximum Mean Discrepancy (MMD)** and **Bayesian Online Change Point Detection (BOCPD)**.

## Experimental Design

Both methods are applied to SPY daily data (2018-2024) in a 2x2 design crossing
detection framework with feature dimensionality:

|  | Univariate (log returns) | Multivariate (log OHLCV) |
|---|---|---|
| **BOCPD** | NIG conjugate model | NIW conjugate model |
| **MMD** | RBF kernel, 1-dim | RBF kernel, 5-dim |

This separates two questions:
- **Framework effect**: nonparametric vs Bayesian on the same features
- **Feature effect**: does multivariate input help or hurt each method?

Results, sensitivity sweeps, and discussion are in
[`notebooks/comparison.md`](notebooks/comparison.md).

## Dependencies

This repo owns no detection logic. It orchestrates three existing packages:

| Package | Role |
|---------|------|
| [finfeatures](https://github.com/whitham-powell/finfeatures) | OHLCV data fetching and feature engineering |
| [mmd-regime-detection](https://github.com/whitham-powell/mmd-regime-detection) | Nonparametric sliding-window MMD detection |
| [bocpd-regime-detection](https://github.com/whitham-powell/bocpd-regime-detection) | Bayesian online change point detection |

## Setup

```bash
git clone https://github.com/whitham-powell/regime-detection-comparison.git
cd regime-detection-comparison
uv sync --extra dev
```

## Usage

```bash
make help
```

```
help      Show this help
sync      Sync .py and .ipynb via jupytext
plots     Execute notebooks and save figures
markdown  Execute notebooks and render as markdown with figures
clean     Remove generated notebooks, markdown, and figures
```

The source of truth is `notebooks/comparison.py` (Jupytext percent format).
The `.ipynb` and `.md` are generated artifacts.

## Adding New Notebooks

To add a new experiment (e.g., exotic feature combinations):

1. Create a new `.py` file in `notebooks/` using Jupytext percent format (`# %%` cells)
2. The shared dependencies (finfeatures, regime-detection, bocpd) are already available
3. `make sync`, `make markdown`, and `make clean` operate on all `notebooks/*.py` / `notebooks/*.ipynb`

## License

MIT
