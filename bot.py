#!/usr/bin/env python3
"""
bot.py — multi-strategy TopstepX AI bot with a PPO trailing exit.

Each bar:  every active strategy detects its mechanical entry  →  grades it with
its own Chronos+XGBoost model  →  the best graded signal (proba ≥ floor) is
taken  →  the PPO policy trails the stop until exit.

    detect (SuperTrend flip / EMA cross)  →  model grades  →  enter  →  PPO trail

Strategies and exit behaviour are configured in config.py / .env. Run:

    pip install -r requirements.txt
    cp .env.example .env          # then fill in your TopstepX credentials
    python bot.py                 # live (places LIVE orders)
    python bot.py --backtest --symbol NQ --start 2026-01-01 --end 2026-03-01
    python bot.py --retrain-exit  # retrain the PPO trailing exit

⚠️  EDUCATIONAL — live mode places LIVE orders. Run it on a practice/evaluation
    account first. NQ 3-min is the models' training scope.
"""
import os
import time

import config
import exit_manager as ex
import strategies as strat
from broker import SIDE, TopstepXClient
from logsetup import get_logger

log = get_logger()


class BotContext:
    """Everything a bar needs: the broker (live or simulated), the active
    strategies, the PPO policy, and the trade identifiers. Shared by the live
    loop and the backtester so they run identical per-bar logic."""

    def __init__(self, client, account_id, contract_id, tick_size,
                 log_candles=True):
        import trail_exit_env as tee     # numpy-only PPO policy loader
        self.client = client
        self.account_id = account_id
        self.contract_id = contract_id
        self.tick_size = tick_size
        self.log_candles = log_candles
        self.tee = tee
        self.strategies = strat.make_strategies()
        self.policy = None
        if config.USE_PPO_EXIT and os.path.exists(config.POLICY_PATH):
            self.policy = tee.NumpyMlpPolicy.load(config.POLICY_PATH)
        self.trailing = bool(self.policy) and config.USE_TRAILING_STOP

    @property
    def exit_mode(self):
        return ("PPO native-trail" if self.trailing else
                "PPO stop-reprice" if self.policy else f"fixed {config.RR}R")


def handle_bar(ctx: BotContext, bars, trade_state):
    """Run one bar of bot logic: if in a trade, trail the stop; otherwise detect
    + grade across strategies and enter the best signal. Returns the updated
    trade_state. Identical for live trading and backtesting."""
    c = ctx.client
    stamp = bars["time"].iloc[-1].strftime("%Y-%m-%d %H:%M")
    if ctx.log_candles:
        last = bars.iloc[-1]
        log.info("candle %s  O=%.2f H=%.2f L=%.2f C=%.2f V=%s", stamp,
                 last["open"], last["high"], last["low"], last["close"],
                 last.get("volume", "?"))

    pos = c.open_position(ctx.account_id, ctx.contract_id)
    if pos:
        # In a trade — let the PPO policy trail the stop. Without a policy the
        # attached fixed bracket manages the exit itself.
        if ctx.policy is None:
            return trade_state
        if trade_state is None:
            trade_state = ex.reconstruct_state(
                c, ctx.account_id, ctx.contract_id, pos, ctx.strategies[0])
        if trade_state:
            line = trade_state["strategy"].reference_line(bars)
            ex.manage_trail(ctx.tee, ctx.policy, c, ctx.account_id,
                            ctx.contract_id, ctx.tick_size, bars, line,
                            trade_state, ctx.trailing)
        return trade_state

    # Flat — detect + grade across every active strategy, take the best.
    candidates = []
    for s in ctx.strategies:
        sig = s.detect(bars)
        if sig is None:
            continue
        sig.proba, sig.r_hat = s.grade(bars, sig)
        side_txt = "LONG" if sig.direction > 0 else "SHORT"
        take = sig.proba >= config.PROBA_FLOOR
        log.info("signal %s [%s] %s | proba=%.3f r_hat=%.2f | %s", stamp,
                 s.name, side_txt, sig.proba, sig.r_hat,
                 "TAKE" if take else f"skip (<{config.PROBA_FLOOR})")
        if take:
            candidates.append((s, sig))

    if not candidates:
        return None
    s, sig = max(candidates, key=lambda c_: c_[1].proba)   # highest proba wins

    stop_ticks = max(1, round(sig.risk / ctx.tick_size))
    side = SIDE["BUY"] if sig.direction > 0 else SIDE["SELL"]
    side_txt = "LONG" if sig.direction > 0 else "SHORT"

    if ctx.policy is not None:
        trade_state = {"sign": sig.direction, "entry": sig.entry,
                       "risk": sig.risk, "stop": sig.stop, "bars_held": 0,
                       "mfe": 0.0, "trail_ticks": stop_ticks, "strategy": s}
        if ctx.trailing:
            c.place_market_with_trail(ctx.account_id, ctx.contract_id,
                                      side=side, size=config.SIZE,
                                      trail_ticks=stop_ticks)
            log.info("🎯 ENTER %s %s [%s] %d | native trail %dt | PPO (proba %.3f)",
                     stamp, side_txt, s.name, config.SIZE, stop_ticks, sig.proba)
        else:
            c.place_market_with_stop(ctx.account_id, ctx.contract_id,
                                     side=side, size=config.SIZE,
                                     stop_ticks=stop_ticks)
            log.info("🎯 ENTER %s %s [%s] %d | stop %dt | PPO reprice (proba %.3f)",
                     stamp, side_txt, s.name, config.SIZE, stop_ticks, sig.proba)
    else:
        target_ticks = max(1, round(config.RR * sig.risk / ctx.tick_size))
        c.place_market_with_brackets(ctx.account_id, ctx.contract_id,
                                     side=side, size=config.SIZE,
                                     stop_ticks=stop_ticks, target_ticks=target_ticks)
        log.info("🎯 ENTER %s %s [%s] %d | stop %dt | target %dt (%sR)",
                 stamp, side_txt, s.name, config.SIZE, stop_ticks, target_ticks, config.RR)
    return trade_state


def run():
    """Live trading loop against the real TopstepX broker."""
    client = TopstepXClient(config.TOPSTEPX_USERNAME, config.TOPSTEPX_API_KEY)
    client.authenticate()
    acct = client.pick_account(config.ACCOUNT)
    contract = client.get_active_contract(config.SYMBOL)
    ctx = BotContext(client, acct["id"], contract["id"],
                     float(contract["tickSize"]))
    names = "+".join(s.name for s in ctx.strategies)
    log.info("✅ %s | %s | %d-min | [%s] | exit: %s", acct["name"],
             ctx.contract_id, config.TIMEFRAME_MIN, names, ctx.exit_mode)
    log.info("▶ running — Ctrl-C to stop")

    trade_state = None
    while True:
        # wait for the next bar close (+2s so the API has published it)
        period = config.TIMEFRAME_MIN * 60
        time.sleep(period - (time.time() % period) + 2)
        try:
            bars = client.get_bars(ctx.contract_id, config.TIMEFRAME_MIN)
            if len(bars) < config.CTX + 30:    # need >=128 closes + warmup
                continue
            trade_state = handle_bar(ctx, bars, trade_state)
        except Exception as e:        # keep the loop alive on transient errors
            log.warning("⚠️  %s", e)


def _retrain_exit(quick: bool, timesteps: int):
    """Retrain the PPO trailing-exit policy (delegates to train_ppo_exit)."""
    import sys
    import train_ppo_exit
    sys.argv = ["train_ppo_exit.py"] + (
        ["--quick"] if quick else ["--timesteps", str(timesteps)])
    log.info("retraining PPO exit (%s)…", "quick" if quick else f"{timesteps} steps")
    train_ppo_exit.main()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="multi-strategy AI futures bot")
    ap.add_argument("--backtest", action="store_true",
                    help="simulate over a local CSV (no API calls)")
    ap.add_argument("--symbol", default=config.SYMBOL,
                    help="backtest symbol (uses data/<symbol>_3min.csv)")
    ap.add_argument("--start", help="backtest start date (YYYY-MM-DD, inclusive)")
    ap.add_argument("--end", help="backtest end date (YYYY-MM-DD, exclusive)")
    ap.add_argument("--retrain-exit", action="store_true",
                    help="retrain the PPO trailing-exit policy, then exit")
    ap.add_argument("--quick", action="store_true",
                    help="with --retrain-exit: fast smoke train")
    ap.add_argument("--timesteps", type=int, default=600_000,
                    help="with --retrain-exit: PPO training steps")
    args = ap.parse_args()

    if args.retrain_exit:
        _retrain_exit(args.quick, args.timesteps)
        raise SystemExit(0)

    if args.backtest:
        import backtest
        backtest.run_backtest(symbol=args.symbol, start=args.start, end=args.end)
        raise SystemExit(0)

    if not config.TOPSTEPX_USERNAME or not config.TOPSTEPX_API_KEY:
        raise SystemExit("missing credentials — copy .env.example to .env and "
                         "set TOPSTEPX_USERNAME / TOPSTEPX_API_KEY")
    run()
