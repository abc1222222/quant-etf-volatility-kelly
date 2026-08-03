# ETF Volatility and Kelly Allocation

This project contains two related ETF strategy research modules:

- `har_yz_return_kelly_etf`: Yang-Zhang volatility estimation, HAR forecasting, return/Kelly overlay, and trend filters.
- `egarch_return_kelly_etf`: EGARCH-based volatility forecasting with return/Kelly allocation logic.

## What It Demonstrates

- Walk-forward time-series modeling with no lookahead in signal generation.
- Volatility forecasting from OHLCV data.
- Long-only position sizing with risk constraints.
- Regime-aware reporting with bull/bear summaries.
- Reproducible research outputs: summary CSV files and selected charts.

## Included

- Strategy source code.
- Small ETF universe files.
- Selected result summaries and charts.

## Excluded

- Full raw market data caches.
- Large parameter dumps.
- Private local data files.
