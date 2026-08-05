from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import baostock as bs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError:  # pragma: no cover - optional fallback
    ak = None


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
UNIVERSE_FILE = ROOT / "etf_universe.csv"
TRADING_DAYS = 252


@dataclass(frozen=True)
class BacktestConfig:
    yz_window: int = 20
    har_week: int = 5
    har_month: int = 22
    min_train: int = 252
    target_vol: float = 0.15
    fee_bps: float = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HAR-YZ ETF volatility backtests.")
    parser.add_argument("--start", default="2015-01-01", help="Download start date.")
    parser.add_argument("--end", default="2026-05-31", help="Download end date.")
    parser.add_argument("--refresh", action="store_true", help="Redownload ETF data.")
    parser.add_argument("--target-vol", type=float, default=0.15, help="Annual target volatility.")
    parser.add_argument("--fee-bps", type=float, default=2.0, help="One-way turnover cost in bps.")
    parser.add_argument("--min-train", type=int, default=252, help="Minimum HAR training rows.")
    return parser.parse_args()


def baostock_code_to_file(code: str) -> Path:
    return DATA_DIR / f"{code.replace('.', '_')}.parquet"


def load_universe() -> pd.DataFrame:
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"Missing universe file: {UNIVERSE_FILE}")
    universe = pd.read_csv(UNIVERSE_FILE)
    required = {"code", "name", "benchmark"}
    missing = required.difference(universe.columns)
    if missing:
        raise ValueError(f"Universe file missing columns: {sorted(missing)}")
    return universe


def download_one_etf(code: str, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = baostock_code_to_file(code)
    if cache_file.exists() and not refresh:
        cached = pd.read_parquet(cache_file)
        if has_enough_history(cached, start):
            return cached

    df = download_one_etf_baostock(code, start, end)
    if not has_enough_history(df, start):
        try:
            df_ak = download_one_etf_sina(code, start, end)
            if len(df_ak) > len(df):
                df = df_ak
        except Exception as exc:
            print(f"sina fallback failed for {code}: {exc}")
            try:
                df_ak = download_one_etf_akshare(code, start, end)
                if len(df_ak) > len(df):
                    df = df_ak
            except Exception as exc2:
                print(f"eastmoney fallback failed for {code}: {exc2}")

    df.to_parquet(cache_file, index=False)
    return df


def has_enough_history(df: pd.DataFrame, start: str, min_rows: int = 500) -> bool:
    if df.empty or "date" not in df.columns:
        return False
    return len(df) >= min_rows


def download_one_etf_baostock(code: str, start: str, end: str) -> pd.DataFrame:
    fields = "date,code,open,high,low,close,volume,amount"
    rs = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag="2",
    )
    rows: list[list[str]] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(f"baostock failed for {code}: {rs.error_msg}")
    if not rows:
        raise RuntimeError(f"No rows returned for {code}")

    df = pd.DataFrame(rows, columns=fields.split(","))
    return normalize_ohlcv(df)


def download_one_etf_akshare(code: str, start: str, end: str) -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("akshare is not installed")
    symbol = code.split(".")[-1]
    last_error: Exception | None = None
    for _ in range(3):
        try:
            raw = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq",
            )
            break
        except Exception as exc:  # network endpoint can be flaky
            last_error = exc
    else:
        raise RuntimeError(f"akshare failed for {code}: {last_error}")

    rename = {
        "日期": "date",
        "开�?: "open",
        "最�?: "high",
        "最�?: "low",
        "收盘": "close",
        "成交�?: "volume",
        "成交�?: "amount",
    }
    df = raw.rename(columns=rename)
    df["code"] = code
    return normalize_ohlcv(df[["date", "code", "open", "high", "low", "close", "volume", "amount"]])


def download_one_etf_sina(code: str, start: str, end: str) -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("akshare is not installed")
    symbol = code.replace(".", "")
    raw = ak.fund_etf_hist_sina(symbol=symbol)
    if raw.empty:
        raise RuntimeError(f"sina returned no rows for {code}")
    df = raw.copy()
    df["code"] = code
    if "amount" not in df.columns:
        df["amount"] = np.nan
    df = normalize_ohlcv(df[["date", "code", "open", "high", "low", "close", "volume", "amount"]])
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    return df.loc[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return df


def yang_zhang_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    overnight = np.log(open_ / prev_close)
    open_close = np.log(close / open_)
    rs = np.log(high / open_) * np.log(high / close) + np.log(low / open_) * np.log(low / close)
    k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))

    yz_var = (
        overnight.rolling(window).var(ddof=1)
        + k * open_close.rolling(window).var(ddof=1)
        + (1.0 - k) * rs.rolling(window).mean()
    )
    return np.sqrt(yz_var.clip(lower=0.0))


def build_har_frame(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    out = trim_after_extreme_price_jumps(df)
    out["yz"] = yang_zhang_volatility(out, cfg.yz_window)
    out["yz_ann"] = out["yz"] * math.sqrt(TRADING_DAYS)
    out["ret1"] = out["close"].pct_change()
    out["log_ret1"] = np.log(out["close"] / out["close"].shift(1))
    out["ma5"] = out["close"].rolling(5).mean()
    out["ma20"] = out["close"].rolling(20).mean()
    out["ret20"] = out["close"] / out["close"].shift(20) - 1.0

    out["har_d"] = out["yz"]
    out["har_w"] = out["yz"].rolling(cfg.har_week).mean()
    out["har_m"] = out["yz"].rolling(cfg.har_month).mean()
    out["target_yz_next"] = out["yz"].shift(-1)
    return out


def trim_after_extreme_price_jumps(df: pd.DataFrame, max_abs_return: float = 0.25) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    ret = out["close"].pct_change()
    bad = ret.abs() > max_abs_return
    if bad.any():
        last_bad_pos = int(np.flatnonzero(bad.to_numpy())[-1])
        out = out.iloc[last_bad_pos:].copy().reset_index(drop=True)
        out.attrs["trimmed_after_extreme_jump"] = str(out.loc[0, "date"])
    else:
        out.attrs["trimmed_after_extreme_jump"] = ""
    return out


def walk_forward_har(frame: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    model_cols = ["har_d", "har_w", "har_m"]
    pred = pd.Series(index=frame.index, dtype=float)
    params = []
    usable = frame.dropna(subset=model_cols + ["target_yz_next"]).copy()
    x_all = np.column_stack([np.ones(len(usable)), usable[model_cols].to_numpy(dtype=float)])
    y_all = usable["target_yz_next"].to_numpy(dtype=float)

    for pos, idx in enumerate(usable.index):
        if pos < cfg.min_train:
            continue
        beta, *_ = np.linalg.lstsq(x_all[:pos], y_all[:pos], rcond=None)
        x_now = np.array([1.0, *frame.loc[idx, model_cols].to_numpy(dtype=float)])
        pred.loc[idx] = float(x_now @ beta)
        params.append(
            {
                "date": frame.loc[idx, "date"],
                "beta0": beta[0],
                "beta_d": beta[1],
                "beta_w": beta[2],
                "beta_m": beta[3],
            }
        )

    out = frame.copy()
    out["pred_yz_next"] = pred.clip(lower=0.0001)
    out["pred_yz_next_ann"] = out["pred_yz_next"] * math.sqrt(TRADING_DAYS)
    out.attrs["har_params"] = pd.DataFrame(params)
    return out


def add_price_bands(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for label, z in [("80", 1.2815515655446004), ("95", 1.959963984540054)]:
        out[f"band{label}_low"] = out["close"] * np.exp(-z * out["pred_yz_next"])
        out[f"band{label}_high"] = out["close"] * np.exp(z * out["pred_yz_next"])
    return out


def strategy_positions(frame: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    out = frame.copy()
    vol_scale = (cfg.target_vol / out["pred_yz_next_ann"]).clip(lower=0.0, upper=1.0)
    trends = {
        "close_gt_ma20": out["close"] > out["ma20"],
        "ma5_gt_ma20": out["ma5"] > out["ma20"],
        "ret20_gt_0": out["ret20"] > 0.0,
    }
    for name, trend in trends.items():
        pos = trend.astype(float) * vol_scale
        out[f"pos_{name}"] = pos.fillna(0.0)
        turnover = out[f"pos_{name}"].diff().abs().fillna(out[f"pos_{name}"].abs())
        cost = turnover * cfg.fee_bps / 10000.0
        out[f"strat_ret_{name}"] = out[f"pos_{name}"].shift(1).fillna(0.0) * out["ret1"] - cost
        out[f"equity_{name}"] = (1.0 + out[f"strat_ret_{name}"].fillna(0.0)).cumprod()
    out["buy_hold_equity"] = (1.0 + out["ret1"].fillna(0.0)).cumprod()
    return out


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    total = float((1.0 + returns).prod())
    years = len(returns) / TRADING_DAYS
    if years <= 0 or total <= 0:
        return np.nan
    return total ** (1.0 / years) - 1.0


def annualized_vol(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series) -> float:
    vol = annualized_vol(returns)
    if not np.isfinite(vol) or vol <= 0:
        return np.nan
    return annualized_return(returns) / vol


def qlike(actual: pd.Series, pred: pd.Series) -> float:
    actual_var = actual.pow(2)
    pred_var = pred.pow(2).clip(lower=1e-10)
    vals = actual_var / pred_var - np.log(actual_var / pred_var).replace([np.inf, -np.inf], np.nan) - 1.0
    return float(vals.replace([np.inf, -np.inf], np.nan).mean())


def summarize(code: str, name: str, benchmark: str, frame: pd.DataFrame) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    valid_pred = frame.dropna(subset=["target_yz_next", "pred_yz_next"])
    pred_stats = {
        "vol_mae": float((valid_pred["target_yz_next"] - valid_pred["pred_yz_next"]).abs().mean()),
        "vol_qlike": qlike(valid_pred["target_yz_next"], valid_pred["pred_yz_next"]),
        "latest_close": float(frame["close"].dropna().iloc[-1]),
        "latest_pred_vol_ann": float(frame["pred_yz_next_ann"].dropna().iloc[-1]) if frame["pred_yz_next_ann"].notna().any() else np.nan,
    }
    bh_ret = frame["ret1"].loc[frame["pred_yz_next"].notna()]
    bh_ann = annualized_return(bh_ret)
    for trend in ["close_gt_ma20", "ma5_gt_ma20", "ret20_gt_0"]:
        ret = frame.loc[frame["pred_yz_next"].notna(), f"strat_ret_{trend}"]
        equity = frame.loc[frame["pred_yz_next"].notna(), f"equity_{trend}"]
        ann_ret = annualized_return(ret)
        rows.append(
            {
                "code": code,
                "name": name,
                "benchmark": benchmark,
                "trend": trend,
                "ann_return": ann_ret,
                "ann_vol": annualized_vol(ret),
                "sharpe": sharpe(ret),
                "max_drawdown": max_drawdown(equity),
                "buy_hold_ann_return": bh_ann,
                "alpha_vs_buy_hold": ann_ret - bh_ann if np.isfinite(ann_ret) and np.isfinite(bh_ann) else np.nan,
                "avg_position": float(frame.loc[frame["pred_yz_next"].notna(), f"pos_{trend}"].mean()),
                **pred_stats,
            }
        )
    return rows


def plot_one(code: str, frame: pd.DataFrame) -> None:
    chart_dir = OUTPUT_DIR / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    plot_df = frame.dropna(subset=["pred_yz_next_ann"]).copy()
    if plot_df.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(plot_df["date"], plot_df["close"], label="close", color="#222222", linewidth=1.2)
    axes[0].fill_between(
        plot_df["date"],
        plot_df["band80_low"],
        plot_df["band80_high"],
        color="#4c78a8",
        alpha=0.18,
        label="80% next-day band",
    )
    axes[0].set_title(f"{code} price and HAR-YZ risk band")
    axes[0].legend(loc="upper left")

    axes[1].plot(plot_df["date"], plot_df["yz_ann"], label="realized YZ ann", color="#f58518", linewidth=1.0)
    axes[1].plot(plot_df["date"], plot_df["pred_yz_next_ann"], label="predicted next YZ ann", color="#54a24b", linewidth=1.0)
    axes[1].axhline(0.15, color="#999999", linewidth=0.8, linestyle="--", label="15% target")
    axes[1].legend(loc="upper left")

    axes[2].plot(plot_df["date"], plot_df["buy_hold_equity"], label="buy hold", color="#777777", linewidth=1.0)
    for trend, color in [("close_gt_ma20", "#4c78a8"), ("ma5_gt_ma20", "#e45756"), ("ret20_gt_0", "#72b7b2")]:
        axes[2].plot(plot_df["date"], plot_df[f"equity_{trend}"], label=trend, linewidth=1.0, color=color)
    axes[2].legend(loc="upper left")
    axes[2].set_title("Walk-forward strategy equity")

    fig.tight_layout()
    fig.savefig(chart_dir / f"{code.replace('.', '_')}.png", dpi=150)
    plt.close(fig)


def run() -> None:
    args = parse_args()
    cfg = BacktestConfig(target_vol=args.target_vol, fee_bps=args.fee_bps, min_train=args.min_train)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = load_universe()

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")

    summary_rows: list[dict[str, float | str]] = []
    latest_rows: list[dict[str, float | str]] = []
    failures: list[dict[str, str]] = []
    try:
        for row in universe.itertuples(index=False):
            code = str(row.code)
            print(f"Processing {code} {row.name} ...")
            try:
                raw = download_one_etf(code, args.start, args.end, args.refresh)
                frame = build_har_frame(raw, cfg)
                frame = walk_forward_har(frame, cfg)
                frame = add_price_bands(frame)
                frame = strategy_positions(frame, cfg)

                out_file = OUTPUT_DIR / f"{code.replace('.', '_')}_signals.csv"
                frame.to_csv(out_file, index=False, encoding="utf-8-sig")
                params = frame.attrs.get("har_params", pd.DataFrame())
                if not params.empty:
                    params.to_csv(OUTPUT_DIR / f"{code.replace('.', '_')}_har_params.csv", index=False, encoding="utf-8-sig")
                plot_one(code, frame)

                summary_rows.extend(summarize(code, row.name, row.benchmark, frame))
                latest_valid = frame.dropna(subset=["pred_yz_next"])
                if latest_valid.empty:
                    failures.append({"code": code, "error": "not enough history for walk-forward HAR"})
                else:
                    latest = latest_valid.iloc[-1]
                    latest_rows.append(
                        {
                            "code": code,
                            "name": row.name,
                            "benchmark": row.benchmark,
                            "date": latest["date"],
                            "close": latest["close"],
                            "pred_daily_vol": latest["pred_yz_next"],
                            "pred_annual_vol": latest["pred_yz_next_ann"],
                            "band80_low": latest["band80_low"],
                            "band80_high": latest["band80_high"],
                            "band95_low": latest["band95_low"],
                            "band95_high": latest["band95_high"],
                            "pos_close_gt_ma20": latest["pos_close_gt_ma20"],
                            "pos_ma5_gt_ma20": latest["pos_ma5_gt_ma20"],
                            "pos_ret20_gt_0": latest["pos_ret20_gt_0"],
                        }
                    )
            except Exception as exc:
                failures.append({"code": code, "error": repr(exc)})
                print(f"FAILED {code}: {exc}")
    finally:
        bs.logout()

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["trend", "sharpe"], ascending=[True, False])
        summary.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    latest_df = pd.DataFrame(latest_rows)
    if not latest_df.empty:
        latest_df.to_csv(OUTPUT_DIR / "latest_signals.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "start": args.start,
        "end": args.end,
        "target_vol": cfg.target_vol,
        "fee_bps": cfg.fee_bps,
        "min_train": cfg.min_train,
        "universe_file": str(UNIVERSE_FILE),
        "failures": failures,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
