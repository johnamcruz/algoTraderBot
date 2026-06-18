#!/usr/bin/env python3
"""backtest.py — run the bot over a local CSV with no API calls.

Drives the exact live per-bar logic (bot.handle_bar) through a simulated broker
(sim_broker.SimBroker), so entries, grading and the PPO trailing exit behave
just like live — only fills come from history instead of the broker.

    python bot.py --backtest --symbol NQ --start 2026-01-01 --end 2026-03-01

Each bar feeds a trailing window (not the whole history) to keep indicator
recomputation cheap; the simulated broker fills/exits against the full series.
"""
import os

import pandas as pd

import config
from logsetup import LOG_DIR, get_logger
from sim_broker import SimBroker

log = get_logger()

WINDOW = 500           # trailing bars handed to each step (indicator warmup + CTX)


def _load(symbol: str, end) -> pd.DataFrame:
    path = os.path.join(config.HERE, "data", f"{symbol}_3min.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no data file: {path}")
    df = pd.read_csv(path).rename(columns={"datetime": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if end:
        df = df[df["time"] < pd.Timestamp(end, tz="UTC")]
    return df.reset_index(drop=True)


def _summary(trades, symbol):
    import numpy as np
    if not trades:
        log.info("no trades in range")
        return
    r = np.array([t.r for t in trades])
    mfe = np.array([t.mfe_r for t in trades])
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    pf = wins / losses if losses > 0 else float("inf")
    capture = r.sum() / mfe.sum() if mfe.sum() > 0 else 0.0
    log.info("─" * 60)
    log.info("BACKTEST %s | trades=%d  win=%.1f%%  meanR=%+.3f  sumR=%+.2f  PF=%.2f",
             symbol, len(r), 100 * (r > 0).mean(), r.mean(), r.sum(), pf)
    log.info("   MFE: mean=%.2fR  max=%.2fR  | capture (sumR/sumMFE)=%.0f%%  "
             "| biggest trend caught=%.2fR", mfe.mean(), mfe.max(),
             100 * capture, r.max())
    # per-strategy and per-exit-reason breakdown
    for key, label in (("strategy", "strategy"), ("reason", "exit")):
        groups = {}
        for t in trades:
            groups.setdefault(getattr(t, key), []).append(t.r)
        for name, rs in groups.items():
            rs = np.array(rs)
            log.info("   %-9s %-10s n=%-4d win=%.0f%% meanR=%+.3f sumR=%+.2f",
                     label, name, len(rs), 100 * (rs > 0).mean(), rs.mean(), rs.sum())
    out = os.path.join(LOG_DIR, f"backtest_{symbol}.csv")
    os.makedirs(LOG_DIR, exist_ok=True)
    pd.DataFrame([t.__dict__ for t in trades]).to_csv(out, index=False)
    log.info("trades written → %s", os.path.relpath(out, config.HERE))


def _resolve_specs(symbol):
    """(tick_size, tick_value) for a backtest — always from the broker API
    (/Contract/search). The API is the single source of truth; there is no
    offline fallback, so credentials are required even for backtests (only the
    bars are mock)."""
    if not (config.TOPSTEPX_USERNAME and config.TOPSTEPX_API_KEY):
        raise SystemExit("contract specs come from the broker API — set "
                         "TOPSTEPX_USERNAME / TOPSTEPX_API_KEY in .env")
    import broker
    ts, tv = broker.fetch_contract_specs(symbol)
    log.info("contract specs (API): %s tick=%g tickValue=$%g", symbol, ts, tv)
    return ts, tv


def drive(ctx, sim, df, start_idx, window=WINDOW):
    """The per-bar backtest loop — the EXACT live logic (bot.handle_bar) driven
    through a SimBroker. Each bar: settle broker exits, then run the bot on a
    trailing window. Returns the recorded trades. Shared by run_backtest and the
    e2e tests so both exercise the same driver."""
    import bot     # imported here to avoid a cycle (bot imports backtest lazily)
    trade_state = None
    for i in range(start_idx, len(df)):
        sim.set_bar(i)
        sim.process_exits()                 # close on stop/target, else trail
        if sim.pos is None:
            trade_state = None              # exited this bar
        win = df.iloc[max(0, i - window + 1): i + 1]
        trade_state = bot.handle_bar(ctx, win, trade_state)
        if sim.pos is not None and trade_state and sim.pos.get("strategy") is None:
            sim.tag_strategy(trade_state["strategy"].name)
    sim.close_open()                        # settle any open trade at the end
    return sim.trades


def run_backtest(symbol="NQ", start=None, end=None):
    import bot     # imported here to avoid a cycle (bot imports backtest lazily)

    if config.base_symbol(symbol) not in config.TRAINED_SYMBOLS:
        log.warning("⚠️  models are trained on %s; %s is out of distribution",
                    "/".join(config.TRAINED_SYMBOLS), symbol)

    # Micros trade their parent's bars; set SYMBOL so feature derivation uses the
    # parent instrument, and load the parent's data file.
    config.SYMBOL = symbol
    base = config.base_symbol(symbol)
    tick, tick_value = _resolve_specs(symbol)
    df = _load(base, end)
    sim = SimBroker(df, tick)
    ctx = bot.BotContext(sim, account_id=0, contract_id=symbol, tick_size=tick,
                         tick_value=tick_value, log_candles=False)

    start_idx = WINDOW
    if start:
        ts = pd.Timestamp(start, tz="UTC")
        hits = df.index[df["time"] >= ts]
        start_idx = max(WINDOW, int(hits[0]) if len(hits) else len(df))

    names = "+".join(s.name for s in ctx.strategies)
    log.info("▶ backtest %s [%s] | %s → %s | %d bars | conf≥%.2f | exit: %s",
             symbol, names,
             start or str(df["time"].iloc[start_idx].date()),
             end or str(df["time"].iloc[-1].date()),
             len(df) - start_idx, config.PROBA_FLOOR, ctx.exit_mode)

    trades = drive(ctx, sim, df, start_idx)
    _summary(trades, symbol)
    return trades
