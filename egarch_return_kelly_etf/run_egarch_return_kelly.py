from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
HAR_ROOT = ROOT.parent / "har_yz_return_kelly_etf"
SOURCE_DATA_DIR = HAR_ROOT / "data" / "raw"
UNIVERSE_FILE = HAR_ROOT / "etf_universe.csv"
OUTPUT_DIR = ROOT / "outputs"
TRADING_DAYS = 252
E_ABS_NORM = math.sqrt(2.0 / math.pi)

TREND_RULES = {
    "close_gt_ma20": lambda df: (df["close"] > df["ma20"]).astype(float),
    "ma5_gt_ma20": lambda df: (df["ma5"] > df["ma20"]).astype(float),
    "ret20_gt_0": lambda df: (df["ret20"] > 0.0).astype(float),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EGARCH return/Kelly ETF strategy.")
    parser.add_argument("--risk-free", type=float, default=0.02)
    parser.add_argument("--min-train", type=int, default=252)
    parser.add_argument("--refit-step", type=int, default=60, help="Refit EGARCH every N trading days.")
    parser.add_argument("--max-position", type=float, default=1.0)
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--bull-bear-ma", type=int, default=200)
    parser.add_argument("--maxiter", type=int, default=300)
    return parser.parse_args()


def file_for_code(code: str) -> Path:
    return SOURCE_DATA_DIR / f"{code.replace('.', '_')}.parquet"


def trim_after_extreme_price_jumps(df: pd.DataFrame, max_abs_return: float = 0.25) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    ret = out["close"].pct_change()
    bad = ret.abs() > max_abs_return
    if bad.any():
        out = out.iloc[int(np.flatnonzero(bad.to_numpy())[-1]) :].copy().reset_index(drop=True)
    return out


def load_etf_frame(code: str) -> pd.DataFrame:
    path = file_for_code(code)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = trim_after_extreme_price_jumps(df.dropna(subset=["open", "high", "low", "close"]))
    df["ret1"] = df["close"].pct_change()
    df["log_ret1"] = np.log(df["close"] / df["close"].shift(1))
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ret20"] = df["close"] / df["close"].shift(20) - 1.0
    return df


def egarch_neg_loglik(params: np.ndarray, returns_pct: np.ndarray) -> float:
    omega, alpha, gamma, beta = params
    r = returns_pct[np.isfinite(returns_pct)]
    if len(r) < 30:
        return 1e12
    var0 = float(np.var(r))
    if not np.isfinite(var0) or var0 <= 1e-8:
        var0 = 1.0
    logh = math.log(var0)
    nll = 0.0
    for i, eps in enumerate(r):
        if i > 0:
            logh = omega + beta * logh + alpha * (abs(z_prev) - E_ABS_NORM) + gamma * z_prev
        logh = float(np.clip(logh, -20.0, 20.0))
        h = math.exp(logh)
        z = eps / math.sqrt(h)
        nll += 0.5 * (math.log(2.0 * math.pi) + logh + eps * eps / h)
        z_prev = float(np.clip(z, -20.0, 20.0))
    if not np.isfinite(nll):
        return 1e12
    return float(nll)


def fit_egarch(returns_pct: np.ndarray, maxiter: int, start_params: np.ndarray | None = None) -> np.ndarray:
    r = returns_pct[np.isfinite(returns_pct)]
    var0 = float(np.var(r)) if len(r) else 1.0
    init = start_params if start_params is not None else np.array([math.log(max(var0, 1e-6)) * 0.05, 0.10, -0.05, 0.90])
    bounds = [(-5.0, 5.0), (-2.0, 2.0), (-2.0, 2.0), (-0.995, 0.995)]
    result = minimize(
        egarch_neg_loglik,
        init,
        args=(r,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-7},
    )
    if result.success and np.isfinite(result.fun):
        return result.x
    return init


def filter_state_and_forecast(params: np.ndarray, returns_pct: np.ndarray) -> tuple[float, float, float]:
    omega, alpha, gamma, beta = params
    r = returns_pct[np.isfinite(returns_pct)]
    var0 = float(np.var(r)) if len(r) > 1 else 1.0
    logh = math.log(max(var0, 1e-6))
    z_prev = 0.0
    for i, eps in enumerate(r):
        if i > 0:
            logh = omega + beta * logh + alpha * (abs(z_prev) - E_ABS_NORM) + gamma * z_prev
        logh = float(np.clip(logh, -20.0, 20.0))
        z_prev = float(np.clip(eps / math.sqrt(math.exp(logh)), -20.0, 20.0))
    next_logh = omega + beta * logh + alpha * (abs(z_prev) - E_ABS_NORM) + gamma * z_prev
    next_logh = float(np.clip(next_logh, -20.0, 20.0))
    return logh, z_prev, next_logh


def walk_forward_egarch_vol(df: pd.DataFrame, min_train: int, refit_step: int, maxiter: int) -> tuple[pd.Series, pd.DataFrame]:
    returns_pct = (df["log_ret1"].astype(float) * 100.0).to_numpy()
    pred = pd.Series(index=df.index, dtype=float)
    params_rows: list[dict[str, float | str]] = []
    last_params: np.ndarray | None = None
    i = min_train
    while i < len(df) - 1:
        train = returns_pct[: i + 1]
        if np.isfinite(train).sum() < min_train:
            i += refit_step
            continue
        params = fit_egarch(train, maxiter=maxiter, start_params=last_params)
        last_params = params
        segment_end = min(i + refit_step, len(df) - 1)
        params_rows.append(
            {
                "date": df.loc[i, "date"],
                "omega": params[0],
                "alpha_abs": params[1],
                "gamma_asym": params[2],
                "beta_persistence": params[3],
            }
        )
        logh, z_prev, next_logh = filter_state_and_forecast(params, returns_pct[: i + 1])
        for j in range(i, segment_end):
            if j == i:
                pred.iloc[j] = math.sqrt(math.exp(next_logh)) / 100.0
                continue
            eps = returns_pct[j]
            if not np.isfinite(eps):
                pred.iloc[j] = math.sqrt(math.exp(next_logh)) / 100.0
                continue
            omega, alpha, gamma, beta = params
            logh = next_logh
            z_prev = float(np.clip(eps / math.sqrt(math.exp(logh)), -20.0, 20.0))
            next_logh = omega + beta * logh + alpha * (abs(z_prev) - E_ABS_NORM) + gamma * z_prev
            next_logh = float(np.clip(next_logh, -20.0, 20.0))
            pred.iloc[j] = math.sqrt(math.exp(next_logh)) / 100.0
        i = segment_end
    return pred.clip(lower=0.0001), pd.DataFrame(params_rows)


def walk_forward_expected_return(df: pd.DataFrame, trend: pd.Series, min_train: int) -> tuple[pd.Series, pd.DataFrame]:
    target = df["ret1"].shift(-1)
    vol = df["pred_egarch_vol_next"].astype(float)
    features = pd.DataFrame(
        {
            "const": 1.0,
            "trend": trend.astype(float),
            "vol": vol,
            "trend_x_vol": trend.astype(float) * vol,
        },
        index=df.index,
    )
    data = pd.concat([features, target.rename("target_next_ret")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    pred = pd.Series(index=df.index, dtype=float)
    params = []
    x_all = data[["const", "trend", "vol", "trend_x_vol"]].to_numpy(dtype=float)
    y_all = data["target_next_ret"].to_numpy(dtype=float)
    for pos, idx in enumerate(data.index):
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


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    total = float((1.0 + returns).prod())
    years = len(returns) / TRADING_DAYS
    return total ** (1.0 / years) - 1.0 if total > 0 and years > 0 else np.nan


def annualized_vol(returns: pd.Series) -> float:
    returns = returns.dropna()
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(returns) > 1 else np.nan


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    eq = (1.0 + returns).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


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


def summarize(code: str, trend_name: str, df: pd.DataFrame, ret_col: str, pos_col: str, risk_free_annual: float, regime: str) -> dict[str, float | str | int]:
    r = df[ret_col].dropna()
    b = df.loc[r.index, "ret1"].astype(float)
    strat_ann = annualized_return(r)
    bench_ann = annualized_return(b)
    strat_vol = annualized_vol(r)
    tracking_error = annualized_vol(r - b)
    alpha, beta, r2 = capm_alpha_beta(r, b, risk_free_annual)
    return {
        "code": code,
        "trend": trend_name,
        "regime": regime,
        "days": int(len(r)),
        "strategy_ann_return": strat_ann,
        "strategy_ann_vol": strat_vol,
        "sharpe_excess_rf": (strat_ann - risk_free_annual) / strat_vol if strat_vol and np.isfinite(strat_vol) else np.nan,
        "max_drawdown": max_drawdown(r),
        "benchmark_ann_return": bench_ann,
        "active_ann_return_vs_bh": strat_ann - bench_ann if np.isfinite(strat_ann) and np.isfinite(bench_ann) else np.nan,
        "tracking_error_ann": tracking_error,
        "information_ratio": (strat_ann - bench_ann) / tracking_error
        if tracking_error and np.isfinite(tracking_error) and np.isfinite(strat_ann) and np.isfinite(bench_ann)
        else np.nan,
        "capm_alpha_ann": alpha,
        "capm_beta": beta,
        "capm_r2": r2,
        "avg_position": float(df.loc[r.index, pos_col].mean()),
    }


def run() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(UNIVERSE_FILE)
    summary_rows = []
    bull_bear_rows = []
    latest_rows = []
    failures = []

    for row in universe.itertuples(index=False):
        code = str(row.code)
        print(f"Processing {code} ...")
        try:
            df = load_etf_frame(code)
            pred_vol, egarch_params = walk_forward_egarch_vol(df, args.min_train, args.refit_step, args.maxiter)
            df["pred_egarch_vol_next"] = pred_vol
            df["pred_egarch_vol_next_ann"] = df["pred_egarch_vol_next"] * math.sqrt(TRADING_DAYS)
            if not egarch_params.empty:
                egarch_params.to_csv(OUTPUT_DIR / f"{code.replace('.', '_')}_egarch_params.csv", index=False, encoding="utf-8-sig")

            return_params_all = []
            for trend_name, trend_fn in TREND_RULES.items():
                trend = trend_fn(df)
                mu, return_params = walk_forward_expected_return(df, trend, args.min_train)
                if not return_params.empty:
                    return_params.insert(0, "trend", trend_name)
                    return_params_all.append(return_params)
                variance = df["pred_egarch_vol_next"].pow(2).clip(lower=1e-8)
                raw_pos = mu / variance
                pos = raw_pos.clip(lower=0.0, upper=args.max_position).fillna(0.0)
                df[f"trend_{trend_name}"] = trend
                df[f"mu_hat_{trend_name}"] = mu
                df[f"kelly_raw_{trend_name}"] = raw_pos
                df[f"position_{trend_name}"] = pos
                turnover = pos.diff().abs().fillna(pos.abs())
                cost = turnover * args.fee_bps / 10000.0
                df[f"strategy_ret_{trend_name}"] = pos.shift(1).fillna(0.0) * df["ret1"] - cost
                df[f"strategy_equity_{trend_name}"] = (1.0 + df[f"strategy_ret_{trend_name}"].fillna(0.0)).cumprod()

                valid = df[df[f"mu_hat_{trend_name}"].notna()].copy()
                if valid.empty:
                    continue
                summary_rows.append(summarize(code, trend_name, valid, f"strategy_ret_{trend_name}", f"position_{trend_name}", args.risk_free, "all"))

                valid["ma_regime"] = valid["close"].rolling(args.bull_bear_ma).mean()
                valid = valid[valid["ma_regime"].notna()].copy()
                valid["regime"] = np.where(valid["close"] > valid["ma_regime"], "bull_close_gt_ma200", "bear_close_le_ma200")
                for regime, group in valid.groupby("regime"):
                    bull_bear_rows.append(
                        summarize(code, trend_name, group, f"strategy_ret_{trend_name}", f"position_{trend_name}", args.risk_free, regime)
                    )

                latest = valid.tail(1)
                if not latest.empty:
                    last = latest.iloc[0]
                    latest_rows.append(
                        {
                            "code": code,
                            "trend": trend_name,
                            "date": last["date"],
                            "close": last["close"],
                            "pred_daily_vol": last["pred_egarch_vol_next"],
                            "pred_annual_vol": last["pred_egarch_vol_next_ann"],
                            "trend_value": last[f"trend_{trend_name}"],
                            "mu_hat_next_day": last[f"mu_hat_{trend_name}"],
                            "kelly_raw": last[f"kelly_raw_{trend_name}"],
                            "position": last[f"position_{trend_name}"],
                        }
                    )

            df.to_csv(OUTPUT_DIR / f"{code.replace('.', '_')}_egarch_return_kelly_signals.csv", index=False, encoding="utf-8-sig")
            if return_params_all:
                pd.concat(return_params_all, ignore_index=True).to_csv(
                    OUTPUT_DIR / f"{code.replace('.', '_')}_return_model_params.csv", index=False, encoding="utf-8-sig"
                )
        except Exception as exc:
            failures.append({"code": code, "error": repr(exc)})
            print(f"FAILED {code}: {exc}")

    summary = pd.DataFrame(summary_rows).sort_values(["information_ratio", "sharpe_excess_rf"], ascending=False)
    bull_bear = pd.DataFrame(bull_bear_rows).sort_values(["regime", "information_ratio"], ascending=[True, False])
    latest = pd.DataFrame(latest_rows).sort_values(["code", "trend"])
    summary.to_csv(OUTPUT_DIR / "egarch_return_kelly_summary.csv", index=False, encoding="utf-8-sig")
    bull_bear.to_csv(OUTPUT_DIR / "egarch_return_kelly_bull_bear_ma200.csv", index=False, encoding="utf-8-sig")
    latest.to_csv(OUTPUT_DIR / "egarch_return_kelly_latest.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(
            {
                "source_data_dir": str(SOURCE_DATA_DIR),
                "risk_free": args.risk_free,
                "min_train": args.min_train,
                "refit_step": args.refit_step,
                "max_position": args.max_position,
                "fee_bps": args.fee_bps,
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {OUTPUT_DIR / 'egarch_return_kelly_summary.csv'}")


if __name__ == "__main__":
    run()
