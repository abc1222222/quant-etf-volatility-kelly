# HAR-YZ ETF volatility strategy

This subfolder is isolated from the existing Alpha158 stock project. It downloads
China broad-based ETF OHLCV data, estimates rolling 20-day Yang-Zhang volatility,
uses a walk-forward HAR model to predict next-day volatility, and backtests
volatility-targeted ETF positions with simple trend filters.

## What it does

- Data: daily ETF OHLCV from baostock, cached under `data/raw`.
- Volatility: 20-day Yang-Zhang daily volatility, also annualized.
- HAR model: `YZ[t+1] = b0 + bd * YZ[t] + bw * mean(YZ[t-4:t]) + bm * mean(YZ[t-21:t])`.
- Walk-forward: each prediction uses only history available at the signal date.
- Trend filters:
  - `close_gt_ma20`: close above 20-day moving average.
  - `ma5_gt_ma20`: 5-day moving average above 20-day moving average.
  - `ret20_gt_0`: past 20-day return positive.
- Position: `trend * min(1, target_vol / predicted_annual_vol)`.
- Risk ranges: next-day 80% and 95% lognormal price bands with zero drift.
- Outputs: per-ETF prediction files, strategy metrics, summary CSV, and charts.

## Run

From the repository root:

```bash
python har_yz_return_kelly_etf/run_har_yz_etf.py --start 2015-01-01 --end 2026-05-31
```

Or from inside this folder:

```bash
cd har_yz_return_kelly_etf && python run_har_yz_etf.py
```

To force a fresh download:

```bash
python har_yz_return_kelly_etf/run_har_yz_etf.py --refresh
```

Main outputs are written to `har_yz_return_kelly_etf/outputs`. See the root
`README.md` for the full output listing and for the Kelly overlay step
(`run_return_kelly_overlay.py`).

## Notes

The ETF list is intentionally small and focused on broad-based ETFs. It is not a
survivorship-bias-free universe. Add or remove rows in `etf_universe.csv` if you
want a different tradable set.
