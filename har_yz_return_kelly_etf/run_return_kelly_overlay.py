from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
TRADING_DAYS = 252
TREND_RULES = {
    "close_gt_ma20": lambda df: (df["close"] > df["ma20"]).astype(float),
    "ma5_gt_ma20": lambda df: (df["ma5"] > df["ma20"]).astype(float),
    "ret20_gt_0": lambda df: (df["ret20"] > 0).astype(float),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Return regression + Kelly overlay on HAR-YZ ETF signals.")
    parser.add_argument("--risk-free", type=float, default=0.02, help="Annual risk-free rate.")
    parser.add_argument("--min-train", type=int, default=252, help="Minimum expanding-window training rows.")
    parser.add_argument("--max-position", type=float, default=1.0, help="Long-only position cap.")
    parser.add_argument("--fee-bps", type=float, default=2.0, help="One-way turnover cost in bps.")
    parser.add_argument("--bull-bear-ma", type=int, default=200, help="MA window for bull/bear split.")
    return parser.parse_args()


def annualized_return(returns: pd.Series, freq: int = TRADING_DAYS) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    total = float((1.0 + returns).prod())
    years = len(returns) / freq
    if total <= 0 or years <= 0:
        return np.nan
    return total ** (1.0 / years) - 1.0


def annualized_vol(returns: pd.Series, freq: int = TRADING_DAYS) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * math.sqrt(freq))


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def capm_alpha_beta(strategy_ret: pd.Series, benchmark_ret: pd.Series, risk_free_annual: float) -> tuple[float, float, float]:
    data = pd.DataFrame({"s": strategy_ret, "b": benchmark_ret}).dropna()
    if len(data) < 30:
        return np.nan, np.nan, np.nan
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS) - 1.0
    y = data["s"].to_numpy(dtype=float) - rf_daily
    x = data["b"].to_numpy(dtype=float) - rf_daily
    xmat = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(xmat, y, rcond=None)
    resid = y - xmat @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else np.nan
    return float(coef[0] * TRADING_DAYS), float(coef[1]), r2


def walk_forward_expected_return(
    df: pd.DataFrame,
    trend: pd.Series,
    min_train: int,
) -> tuple[pd.Series, pd.DataFrame]:
    y = df["ret1"].shift(-1)
    vol = df["pred_yz_next"].astype(float)
    x = pd.DataFrame(
        {
            "const": 1.0,
            "trend": trend.astype(float),
            "vol": vol,
            "trend_x_vol": trend.astype(float) * vol,
        },
        index=df.index,
    )
    data = pd.concat([x, y.rename("target_next_ret")], axis=1).replace([np.inf, -np.inf], np.nan)
    usable = data.dropna()
    pred = pd.Series(index=df.index, dtype=float)
    params = []

    x_all = usable[["const", "trend", "vol", "trend_x_vol"]].to_numpy(dtype=float)
    y_all = usable["target_next_ret"].to_numpy(dtype=float)
    for pos, idx in enumerate(usable.index):
        if pos < min_train:
            continue
        coef, *_ = np.linalg.lstsq(x_all[:pos], y_all[:pos], rcond=None)
        pred.loc[idx] = float(x_all[pos] @ coef)
        params.append(
            {
                "date": df.loc[idx, "date"],
                "beta_const": coef[0],
                "beta_trend": coef[1],
                "beta_vol": coef[2],
                "beta_trend_x_vol": coef[3],
            }
        )
    return pred, pd.DataFrame(params)


def add_strategy(df: pd.DataFrame, trend_name: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    trend = TREND_RULES[trend_name](out)
    mu, params = walk_forward_expected_return(out, trend, args.min_train)
    variance = out["pred_yz_next"].astype(float).pow(2).clip(lower=1e-8)
    raw_position = mu / variance

    # Long-only ETF implementation: negative expected returns go to cash, positives are capped.
    position = raw_position.clip(lower=0.0, upper=args.max_position)
    out[f"trend_{trend_name}"] = trend
    out[f"mu_hat_{trend_name}"] = mu
    out[f"kelly_raw_{trend_name}"] = raw_position
    out[f"kelly_pos_{trend_name}"] = position.fillna(0.0)
    turnover = out[f"kelly_pos_{trend_name}"].diff().abs().fillna(out[f"kelly_pos_{trend_name}"].abs())
    cost = turnover * args.fee_bps / 10000.0
    out[f"kelly_ret_{trend_name}"] = out[f"kelly_pos_{trend_name}"].shift(1).fillna(0.0) * out["ret1"] - cost
    out[f"kelly_equity_{trend_name}"] = (1.0 + out[f"kelly_ret_{trend_name}"].fillna(0.0)).cumprod()
    return out, params


def summarize_block(
    code: str,
    trend_name: str,
    df: pd.DataFrame,
    strategy_col: str,
    pos_col: str,
    risk_free_annual: float,
    regime: str = "all",
) -> dict[str, float | str | int]:
    strategy = df[strategy_col].dropna()
    benchmark = df.loc[strategy.index, "ret1"].astype(float)
    strat_ann = annualized_return(strategy)
    bench_ann = annualized_return(benchmark)
    strat_vol = annualized_vol(strategy)
    tracking_error = annualized_vol(strategy - benchmark)
    capm_alpha, capm_beta, capm_r2 = capm_alpha_beta(strategy, benchmark, risk_free_annual)
    return {
        "code": code,
        "trend": trend_name,
        "regime": regime,
        "days": int(len(strategy)),
        "strategy_ann_return": strat_ann,
        "strategy_ann_vol": strat_vol,
        "sharpe_excess_rf": (strat_ann - risk_free_annual) / strat_vol if strat_vol and np.isfinite(strat_vol) else np.nan,
        "max_drawdown": max_drawdown(strategy),
        "benchmark_ann_return": bench_ann,
        "active_ann_return_vs_bh": strat_ann - bench_ann if np.isfinite(strat_ann) and np.isfinite(bench_ann) else np.nan,
        "tracking_error_ann": tracking_error,
        "information_ratio": (strat_ann - bench_ann) / tracking_error
        if tracking_error and np.isfinite(tracking_error) and np.isfinite(strat_ann) and np.isfinite(bench_ann)
        else np.nan,
        "capm_alpha_ann": capm_alpha,
        "capm_beta": capm_beta,
        "capm_r2": capm_r2,
        "avg_position": float(df.loc[strategy.index, pos_col].mean()),
    }


def run() -> None:
    args = parse_args()
    summary_rows = []
    bull_bear_rows = []
    latest_rows = []
    params_dir = OUTPUT_DIR / "return_kelly_params"
    params_dir.mkdir(parents=True, exist_ok=True)

    for file in sorted(OUTPUT_DIR.glob("*_signals.csv")):
        if file.name == "latest_signals.csv":
            continue
        code = file.stem.replace("_signals", "").replace("_", ".")
        print(f"Processing {code} ...")
        base = pd.read_csv(file, parse_dates=["date"])
        if "pred_yz_next" not in base.columns:
            continue
        base = base.replace([np.inf, -np.inf], np.nan)
        enriched = base.copy()
        all_params = []
        for trend_name in TREND_RULES:
            enriched, params = add_strategy(enriched, trend_name, args)
            if not params.empty:
                params.insert(0, "trend", trend_name)
                all_params.append(params)

            valid = enriched[enriched[f"mu_hat_{trend_name}"].notna()].copy()
            if valid.empty:
                continue
            summary_rows.append(
                summarize_block(
                    code,
                    trend_name,
                    valid,
                    f"kelly_ret_{trend_name}",
                    f"kelly_pos_{trend_name}",
                    args.risk_free,
                    "all",
                )
            )

            valid["ma_regime"] = valid["close"].rolling(args.bull_bear_ma).mean()
            valid = valid[valid["ma_regime"].notna()].copy()
            valid["regime"] = np.where(valid["close"] > valid["ma_regime"], "bull_close_gt_ma200", "bear_close_le_ma200")
            for regime, group in valid.groupby("regime"):
                bull_bear_rows.append(
                    summarize_block(
                        code,
                        trend_name,
                        group,
                        f"kelly_ret_{trend_name}",
                        f"kelly_pos_{trend_name}",
                        args.risk_free,
                        regime,
                    )
                )

            latest = enriched[enriched[f"mu_hat_{trend_name}"].notna()].tail(1)
            if not latest.empty:
                row = latest.iloc[0]
                latest_rows.append(
                    {
                        "code": code,
                        "trend": trend_name,
                        "date": row["date"],
                        "close": row["close"],
                        "pred_daily_vol": row["pred_yz_next"],
                        "pred_variance": row["pred_yz_next"] ** 2,
                        "trend_value": row[f"trend_{trend_name}"],
                        "mu_hat_next_day": row[f"mu_hat_{trend_name}"],
                        "kelly_raw": row[f"kelly_raw_{trend_name}"],
                        "position": row[f"kelly_pos_{trend_name}"],
                    }
                )

        enriched.to_csv(OUTPUT_DIR / f"{code.replace('.', '_')}_return_kelly_signals.csv", index=False, encoding="utf-8-sig")
        if all_params:
            pd.concat(all_params, ignore_index=True).to_csv(
                params_dir / f"{code.replace('.', '_')}_return_model_params.csv",
                index=False,
                encoding="utf-8-sig",
            )

    summary = pd.DataFrame(summary_rows).sort_values(["sharpe_excess_rf", "information_ratio"], ascending=False)
    bull_bear = pd.DataFrame(bull_bear_rows).sort_values(["regime", "sharpe_excess_rf"], ascending=[True, False])
    latest = pd.DataFrame(latest_rows).sort_values(["code", "trend"])
    summary.to_csv(OUTPUT_DIR / "return_kelly_summary.csv", index=False, encoding="utf-8-sig")
    bull_bear.to_csv(OUTPUT_DIR / "return_kelly_bull_bear_ma200.csv", index=False, encoding="utf-8-sig")
    latest.to_csv(OUTPUT_DIR / "return_kelly_latest.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {OUTPUT_DIR / 'return_kelly_summary.csv'}")
    print(f"Saved {OUTPUT_DIR / 'return_kelly_bull_bear_ma200.csv'}")
    print(f"Saved {OUTPUT_DIR / 'return_kelly_latest.csv'}")


if __name__ == "__main__":
    run()
