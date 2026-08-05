# EGARCH ETF return/Kelly strategy

This folder is the EGARCH counterpart to `har_yz_return_kelly_etf`.

It reuses the ETF OHLCV parquet data cached by the HAR module at:

`../har_yz_return_kelly_etf/data/raw`

so `run_har_yz_etf.py` must be run at least once first.

Workflow:

1. Estimate next-day volatility with an expanding-window EGARCH(1,1) model.
2. Build trend features: `close > MA20`, `MA5 > MA20`, `ret20 > 0`.
3. Predict next-day return with:

```text
expected_return = f(trend, vol_hat, trend * vol_hat)
```

4. Set long-only position:

```text
position = clip(expected_return / predicted_variance, 0, 1)
```

5. Report full-sample and MA200 bull/bear metrics.

Run from the repository root:

```bash
python egarch_return_kelly_etf/run_egarch_return_kelly.py
```

Or from inside this folder:

```bash
cd egarch_return_kelly_etf && python run_egarch_return_kelly.py
```

Outputs are written to `egarch_return_kelly_etf/outputs`. See the root `README.md`
for the full output listing.
