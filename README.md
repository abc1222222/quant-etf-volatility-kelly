# ETF Volatility Forecasting and Kelly Allocation

Two volatility-forecasting modules for ETF trading with Kelly-criterion position sizing:

1. **HAR-YZ** — Yang-Zhang volatility estimation + HAR (Heterogeneous Autoregressive) forecasting
2. **EGARCH** — Exponential GARCH(1,1) volatility forecasting with walk-forward refitting

Both modules share a common Kelly overlay pipeline: walk-forward return regression → mu/sigma position sizing → trend filter → CAPM attribution.

## Strategy Architecture

```
ETF OHLCV Data (BaoStock / akshare)
         │
         ▼
┌─────────────────────────┐     ┌──────────────────────────┐
│  har_yz_return_kelly_etf │     │  egarch_return_kelly_etf  │
│                         │     │                          │
│  Yang-Zhang Volatility  │     │  EGARCH(1,1) Volatility   │
│  σ²_YZ = σ²_overnight   │     │  log(σ²_t) = ω + α|ε|/σ  │
│    + k·σ²_OC            │     │    + γ·ε/σ + β·log(σ²)   │
│    + (1-k)·σ²_RS        │     │                          │
│                         │     │  L-BFGS-B + warm start   │
│  HAR Forecast:          │     │  Refit every 60 days     │
│  σ_t = c + d·σ_d        │     │                          │
│      + w·σ_w + m·σ_m   │     │                          │
│  (expanding window OLS) │     │                          │
└───────────┬─────────────┘     └────────────┬─────────────┘
            │                                │
            └────────────┬───────────────────┘
                         ▼
              Kelly Overlay Pipeline:
              ┌──────────────────────────┐
              │ Walk-forward return OLS  │
              │ → μ forecast             │
              │ → Kelly fraction = μ/σ²  │
              │ → Long-only cap          │
              │ → Trend filter (MA rules)│
              │ → Volatility targeting   │
              └──────────┬───────────────┘
                         ▼
              Performance Attribution:
              • CAPM alpha/beta (OLS)
              • Bull/bear split (MA200)
              • Sharpe, max DD, turnover
              • QLIKE volatility eval
```

## Modules

### HAR-YZ Module (`har_yz_return_kelly_etf/`)

| File | Description |
|------|-------------|
| `run_har_yz_etf.py` | Yang-Zhang volatility + HAR forecast + volatility-target backtest |
| `run_return_kelly_overlay.py` | Walk-forward return regression + Kelly position sizing + trend filter |
| `etf_universe.csv` | ETF universe with benchmark classification |
| `outputs/` | Summary CSVs, charts (4 PNG), and JSON config |

### EGARCH Module (`egarch_return_kelly_etf/`)

| File | Description |
|------|-------------|
| `run_egarch_return_kelly.py` | EGARCH(1,1) fitting + Kelly overlay + regime analysis |
| `outputs/` | Summary CSVs and performance attribution |

## Tech Stack

- **Python 3.10+**
- **Data**: baostock, akshare (ETF OHLCV)
- **Optimization**: scipy.optimize (L-BFGS-B for EGARCH MLE)
- **Analysis**: pandas, numpy, matplotlib
- **Storage**: Parquet for cached market data

## Installation

```bash
git clone https://github.com/wangwang11111222/quant-etf-volatility-kelly.git
cd quant-etf-volatility-kelly
pip install -r requirements.txt
```

## Usage

### HAR-YZ Strategy

```bash
cd har_yz_return_kelly_etf

# Download data + run HAR-YZ backtest
python run_har_yz_etf.py --start 2015-01-01 --end 2026-05-31

# Run Kelly overlay on HAR signals
python run_return_kelly_overlay.py --risk-free 0.02 --fee-bps 2.0
```

### EGARCH Strategy

```bash
cd egarch_return_kelly_etf

# Run EGARCH(1,1) backtest (uses HAR module's cached data)
python run_egarch_return_kelly.py --risk-free 0.02 --refit-step 60 --maxiter 300
```

## Key Algorithms

| Algorithm | Formula | Description |
|-----------|---------|-------------|
| Yang-Zhang Volatility | σ²_YZ = σ²_ON + k·σ²_OC + (1-k)·σ²_RS | Overnight + open-close + Rogers-Satchell decomposition |
| HAR Forecast | σ_t = c + d·σ_d + w·σ_w + m·σ_m | Daily, weekly, monthly heterogeneous components |
| EGARCH(1,1) | log(σ²_t) = ω + α·\|ε_{t-1}\|/σ + γ·ε/σ + β·log(σ²) | Asymmetric leverage effect |
| Kelly Criterion | f* = μ / σ² | Optimal fraction for geometric growth |
| Volatility Targeting | w = σ_target / σ_forecast | Scale position to target annual volatility |
| CAPM Regression | r_s - r_f = α + β(r_b - r_f) + ε | OLS alpha/beta with excess returns |

## Configuration

### HAR-YZ (`BacktestConfig`)

```python
BacktestConfig(
    yz_window=20,     # Yang-Zhang estimation window
    har_week=5,       # Weekly HAR component (5 days)
    har_month=22,     # Monthly HAR component (22 days)
    min_train=252,    # Minimum training period
    target_vol=0.15,  # Annual target volatility (15%)
    fee_bps=2.0,      # One-way transaction cost
)
```

### EGARCH

```bash
--refit-step 60    # Refit EGARCH every 60 trading days
--maxiter 300      # L-BFGS-B max iterations
--max-position 1.0 # Long-only position cap
```

## Outputs

| File | Description |
|------|-------------|
| `summary.csv` | Per-ETF performance (return, vol, Sharpe, max DD) |
| `capm_results.csv` | Alpha, beta, R² vs. benchmark |
| `regime_comparison.csv` | Bull/bear regime performance split |
| `*.png` | Cumulative return and drawdown charts |
| `config.json` | Run configuration snapshot |

## License

MIT
