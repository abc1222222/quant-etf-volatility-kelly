# ETF Volatility Forecasting and Kelly Allocation

Two volatility-forecasting modules for China broad-based ETFs, each paired with a
walk-forward return regression and Kelly-style position sizing. The HAR-YZ module
estimates Yang-Zhang volatility and forecasts it with a HAR model; the EGARCH module
fits an EGARCH(1,1) with periodic refitting and reuses the HAR module's cached price
data for a like-for-like comparison.

## Key Features

- **Yang-Zhang volatility + HAR forecast** — 20-day Yang-Zhang estimator, walk-forward
  HAR regression on daily/weekly/monthly components, volatility-targeted positions.
- **EGARCH(1,1) forecast** — expanding-window maximum likelihood via
  `scipy.optimize.minimize` (L-BFGS-B), refit every `--refit-step` trading days.
- **Kelly overlay** — walk-forward OLS return forecast, long-only position
  `clip(expected_return / predicted_variance, 0, max_position)`.
- **Trend filters** — `close_gt_ma20`, `ma5_gt_ma20`, `ret20_gt_0`.
- **Cost-aware reporting** — one-way turnover fees in bps, annualized return/vol,
  Sharpe, max drawdown, CAPM alpha/beta, tracking error, information ratio.
- **Regime split** — bull/bear breakdown using a 200-day moving average.
- **Reproducibility** — each run writes a `run_metadata.json` snapshot of its settings.

## Directory Structure

```
quant-etf-volatility-kelly/
├── har_yz_return_kelly_etf/
│   ├── run_har_yz_etf.py                    # Yang-Zhang volatility + HAR forecast + vol-target backtest
│   ├── run_return_kelly_overlay.py          # Walk-forward return regression + Kelly overlay
│   ├── etf_universe.csv                     # ETF universe (code, name, benchmark, source_note)
│   ├── README.md
│   └── outputs/
│       ├── strategy_summary.csv
│       ├── latest_signals.csv
│       ├── return_kelly_summary.csv
│       ├── return_kelly_latest.csv
│       ├── return_kelly_bull_bear_ma200.csv
│       ├── risk_adjusted_metrics.csv
│       ├── bull_bear_metrics_ma200.csv
│       ├── run_metadata.json
│       └── charts/
│           ├── sh_510050.png
│           ├── sh_510300.png
│           ├── sh_510500.png
│           └── sh_512100.png
├── egarch_return_kelly_etf/
│   ├── run_egarch_return_kelly.py           # EGARCH(1,1) fit + Kelly overlay + HAR comparison
│   ├── README.md
│   └── outputs/
│       ├── egarch_return_kelly_summary.csv
│       ├── egarch_return_kelly_latest.csv
│       ├── egarch_return_kelly_bull_bear_ma200.csv
│       ├── har_yz_vs_egarch_summary.csv
│       ├── har_yz_vs_egarch_bull_bear_ma200.csv
│       └── run_metadata.json
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended. Market data is downloaded through BaoStock (with an
optional akshare fallback) and cached as Parquet under
`har_yz_return_kelly_etf/data/raw/`, which is gitignored.

## Usage

Run each script from the repository root:

```bash
# 1) Download data and run the HAR-YZ volatility backtest
python har_yz_return_kelly_etf/run_har_yz_etf.py --start 2015-01-01 --end 2026-05-31

# Force a fresh download
python har_yz_return_kelly_etf/run_har_yz_etf.py --refresh

# 2) Apply the return regression + Kelly overlay to the HAR-YZ signals
python har_yz_return_kelly_etf/run_return_kelly_overlay.py --risk-free 0.02 --fee-bps 2.0

# 3) Run the EGARCH counterpart (reuses the HAR module's cached data)
python egarch_return_kelly_etf/run_egarch_return_kelly.py --refit-step 60 --maxiter 300
```

Step 1 must run before steps 2 and 3: the overlay reads the per-ETF `*_signals.csv`
files written by `run_har_yz_etf.py`, and the EGARCH script reads
`har_yz_return_kelly_etf/data/raw/` and `har_yz_return_kelly_etf/etf_universe.csv`.

### Command-line options

`run_har_yz_etf.py`

| Option | Default | Description |
|--------|---------|-------------|
| `--start` | `2015-01-01` | Download start date |
| `--end` | `2026-05-31` | Download end date |
| `--refresh` | off | Re-download ETF data |
| `--target-vol` | `0.15` | Annual target volatility |
| `--fee-bps` | `2.0` | One-way turnover cost in bps |
| `--min-train` | `252` | Minimum HAR training rows |

`run_return_kelly_overlay.py`

| Option | Default | Description |
|--------|---------|-------------|
| `--risk-free` | `0.02` | Annual risk-free rate |
| `--min-train` | `252` | Minimum expanding-window training rows |
| `--max-position` | `1.0` | Long-only position cap |
| `--fee-bps` | `2.0` | One-way turnover cost in bps |
| `--bull-bear-ma` | `200` | MA window for the bull/bear split |

`run_egarch_return_kelly.py`

| Option | Default | Description |
|--------|---------|-------------|
| `--risk-free` | `0.02` | Annual risk-free rate |
| `--min-train` | `252` | Minimum training rows |
| `--refit-step` | `60` | Refit EGARCH every N trading days |
| `--max-position` | `1.0` | Long-only position cap |
| `--fee-bps` | `2.0` | One-way turnover cost in bps |
| `--bull-bear-ma` | `200` | MA window for the bull/bear split |
| `--maxiter` | `300` | L-BFGS-B maximum iterations |

## Outputs

Written by `run_har_yz_etf.py` into `har_yz_return_kelly_etf/outputs/`:

| File | Description |
|------|-------------|
| `strategy_summary.csv` | Per-ETF, per-trend-rule backtest metrics |
| `latest_signals.csv` | Most recent signal row for each ETF |
| `<code>_signals.csv` | Full per-ETF signal history (one file per ETF) |
| `<code>_har_params.csv` | Walk-forward HAR coefficients |
| `charts/<code>.png` | Per-ETF chart |
| `run_metadata.json` | Run configuration snapshot |

Written by `run_return_kelly_overlay.py` into the same directory:

| File | Description |
|------|-------------|
| `return_kelly_summary.csv` | Kelly-overlay performance per ETF and trend rule |
| `return_kelly_bull_bear_ma200.csv` | Bull/bear split of the overlay results |
| `return_kelly_latest.csv` | Latest overlay signal per ETF |
| `<code>_return_kelly_signals.csv` | Enriched per-ETF signal history |

Written by `run_egarch_return_kelly.py` into `egarch_return_kelly_etf/outputs/`:

| File | Description |
|------|-------------|
| `egarch_return_kelly_summary.csv` | EGARCH strategy performance per ETF and trend rule |
| `egarch_return_kelly_bull_bear_ma200.csv` | Bull/bear split |
| `egarch_return_kelly_latest.csv` | Latest EGARCH signal per ETF |
| `har_yz_vs_egarch_summary.csv` | HAR-YZ vs. EGARCH side-by-side comparison |
| `har_yz_vs_egarch_bull_bear_ma200.csv` | HAR-YZ vs. EGARCH by regime |
| `<code>_egarch_params.csv` | Fitted EGARCH parameters per refit |
| `run_metadata.json` | Run configuration snapshot |

## Dependencies

From `requirements.txt`: `pandas`, `numpy`, `scipy`, `baostock`, `akshare`,
`matplotlib`, `pyarrow`.

## Notes

- The committed `outputs/` folders are sample results from earlier runs, kept so the
  repository is readable without re-running the pipeline. `risk_adjusted_metrics.csv`
  and `bull_bear_metrics_ma200.csv` come from an earlier iteration of the HAR module
  and are not regenerated by the current scripts.
- The `run_metadata.json` files record the absolute paths of the machine that produced
  them; they are historical artifacts and are overwritten on the next run.
- The ETF universe is intentionally small and focused on broad-based ETFs. It is not
  survivorship-bias free. Edit `har_yz_return_kelly_etf/etf_universe.csv` to change
  the tradable set.
- Two sibling repositories build on this one:
  [`quant-etf-correlation-rotation`](../quant-etf-correlation-rotation) and
  [`quant-etf-cross-sectional-selection`](../quant-etf-cross-sectional-selection).
- Research code only; nothing here is investment advice.

## License

Released under the [MIT License](LICENSE).
