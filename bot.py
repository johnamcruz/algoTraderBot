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
import datetime as dt
import os
import time

import config
import exit_manager as ex
import strategies as strat
from broker import SIDE, make_broker
from logsetup import get_logger

log = get_logger()


class BotContext:
    """Everything a bar needs: the broker (live or simulated), the active
    strategies, the PPO policy, and the trade identifiers. Shared by the live
    loop and the backtester so they run identical per-bar logic."""

    def __init__(self, client, account_id, contract_id, tick_size,
                 tick_value=0.0, log_candles=True):
        import trail_exit_env as tee     # numpy-only PPO policy loader
        self.client = client
        self.account_id = account_id
        self.contract_id = contract_id
        self.tick_size = tick_size
        self.tick_value = tick_value      # $ per tick per contract (for risk sizing)
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

    @property
    def sizing_mode(self):
        if config.RISK_PER_TRADE and self.tick_value:
            return f"risk ${config.RISK_PER_TRADE:g}/trade (≤{config.MAX_CONTRACTS})"
        return f"fixed {config.SIZE}"


def position_size(ctx: BotContext, stop_ticks: int) -> int:
    """Contracts for a trade: risk-based when RISK_PER_TRADE > 0 (size from the
    stop distance), else the fixed SIZE — capped at MAX_CONTRACTS.

        size = min(MAX_CONTRACTS,
                   risk_sizing and stop_ticks
                       ? max(1, floor(RISK_PER_TRADE / (|stop_ticks| × tick_value)))
                       : SIZE)
    """
    if config.RISK_PER_TRADE and ctx.tick_value and stop_ticks:
        per_contract = abs(stop_ticks) * ctx.tick_value
        n = max(1, int(config.RISK_PER_TRADE // per_contract))
    else:
        n = config.SIZE
    return min(config.MAX_CONTRACTS, n)


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
            ex.manage_trail(ctx.tee, ctx.policy, c, ctx.account_id,
                            ctx.contract_id, ctx.tick_size, bars,
                            trade_state, ctx.trailing)
        return trade_state

    # Flat — detect across strategies (cheap), then grade. Strategies that fire
    # on this bar share one Chronos embedding (same context) — computed once.
    fired = [(s, sig) for s in ctx.strategies if (sig := s.detect(bars))]
    candidates = []
    if fired:
        emb = strat.embed_context(bars, len(bars) - 1)   # one Chronos pass per bar
        for s, sig in fired:
            sig.proba, sig.r_hat = s.grade(bars, sig, emb=emb)
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
    size = position_size(ctx, stop_ticks)
    side = SIDE["BUY"] if sig.direction > 0 else SIDE["SELL"]
    side_txt = "LONG" if sig.direction > 0 else "SHORT"

    if ctx.policy is not None:
        trade_state = {"sign": sig.direction, "entry": sig.entry,
                       "risk": sig.risk, "stop": sig.stop, "bars_held": 0,
                       "mfe": 0.0, "trail_ticks": stop_ticks, "strategy": s}
        if ctx.trailing:
            c.place_market_with_trail(ctx.account_id, ctx.contract_id,
                                      side=side, size=size, trail_ticks=stop_ticks)
            log.info("🎯 ENTER %s %s [%s] %d | native trail %dt | PPO (proba %.3f)",
                     stamp, side_txt, s.name, size, stop_ticks, sig.proba)
        else:
            c.place_market_with_stop(ctx.account_id, ctx.contract_id,
                                     side=side, size=size, stop_ticks=stop_ticks)
            log.info("🎯 ENTER %s %s [%s] %d | stop %dt | PPO reprice (proba %.3f)",
                     stamp, side_txt, s.name, size, stop_ticks, sig.proba)
    else:
        target_ticks = max(1, round(config.RR * sig.risk / ctx.tick_size))
        c.place_market_with_brackets(ctx.account_id, ctx.contract_id,
                                     side=side, size=size,
                                     stop_ticks=stop_ticks, target_ticks=target_ticks)
        log.info("🎯 ENTER %s %s [%s] %d | stop %dt | target %dt (%sR)",
                 stamp, side_txt, s.name, size, stop_ticks, target_ticks, config.RR)
    return trade_state


def run():
    """Live trading loop against the configured broker."""
    client = make_broker()
    client.authenticate()
    acct = client.pick_account(config.ACCOUNT)
    contract = client.get_active_contract(config.SYMBOL)
    # tick size / value come straight from the broker contract — never hardcoded.
    tick_size = float(contract["tickSize"])
    tick_value = float(contract["tickValue"])
    ctx = BotContext(client, acct["id"], contract["id"], tick_size, tick_value)
    names = "+".join(s.name for s in ctx.strategies)
    log.info("✅ %s | %s | %d-min | [%s] | conf≥%.2f | exit: %s | size: %s",
             acct["name"], ctx.contract_id, config.TIMEFRAME_MIN, names,
             config.PROBA_FLOOR, ctx.exit_mode, ctx.sizing_mode)
    log.info("▶ running — Ctrl-C to stop")

    trade_state = None
    rolled_on = dt.datetime.now(dt.timezone.utc).date()
    while True:
        # wait for the next bar close (+2s so the API has published it)
        period = config.TIMEFRAME_MIN * 60
        time.sleep(period - (time.time() % period) + 2)
        try:
            # Follow the roll: the broker API is the source of truth for the
            # front month. Re-resolve once a day while flat so a long-running
            # session moves to the new front contract (and its clean warmup
            # history) without a restart.
            today = dt.datetime.now(dt.timezone.utc).date()
            if today != rolled_on and trade_state is None \
                    and client.open_position(ctx.account_id, ctx.contract_id) is None:
                rolled_on = today
                front = client.get_active_contract(config.SYMBOL)
                if front["id"] != ctx.contract_id:
                    ctx.contract_id = front["id"]
                    ctx.tick_size = float(front["tickSize"])
                    ctx.tick_value = float(front["tickValue"])
                    log.info("🔄 rolled to front contract %s (tick %g, $%g/tick)",
                             front.get("name", front["id"]),
                             ctx.tick_size, ctx.tick_value)

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
    ap.add_argument("--strategy", nargs="+", metavar="NAME",
                    choices=list(strat.REGISTRY),
                    help="strategies to run: %(choices)s "
                         "(overrides config.ACTIVE_STRATEGIES)")
    ap.add_argument("--start", help="backtest start date (YYYY-MM-DD, inclusive)")
    ap.add_argument("--end", help="backtest end date (YYYY-MM-DD, exclusive)")
    ap.add_argument("--size", type=int,
                    help="fixed contracts per trade (overrides config.SIZE)")
    ap.add_argument("--risk", type=float,
                    help="$ risk per trade; sizes contracts from the stop "
                         "(overrides config.RISK_PER_TRADE). Use instead of --size")
    ap.add_argument("--max-contracts", type=int,
                    help="cap on risk-sized contracts (overrides config.MAX_CONTRACTS)")
    ap.add_argument("--proba-floor", type=float,
                    help="minimum entry confidence (proba) to take a trade, 0–1 "
                         "(overrides config.PROBA_FLOOR)")
    ap.add_argument("--retrain-exit", action="store_true",
                    help="retrain the PPO trailing-exit policy, then exit")
    ap.add_argument("--quick", action="store_true",
                    help="with --retrain-exit: fast smoke train")
    ap.add_argument("--timesteps", type=int, default=600_000,
                    help="with --retrain-exit: PPO training steps")
    args = ap.parse_args()

    if args.size is not None and args.risk is not None:
        raise SystemExit("use either --size or --risk, not both")
    if args.strategy:
        config.ACTIVE_STRATEGIES = args.strategy
    if args.proba_floor is not None:
        if not 0.0 <= args.proba_floor <= 1.0:
            raise SystemExit("--proba-floor must be between 0 and 1")
        config.PROBA_FLOOR = args.proba_floor
    if args.max_contracts is not None:
        config.MAX_CONTRACTS = args.max_contracts
    if args.size is not None:
        if args.size < 1:
            raise SystemExit("--size must be >= 1")
        config.SIZE = args.size
        config.RISK_PER_TRADE = 0.0      # explicit fixed size disables risk sizing
    if args.risk is not None:
        if args.risk <= 0:
            raise SystemExit("--risk must be > 0")
        config.RISK_PER_TRADE = args.risk

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
