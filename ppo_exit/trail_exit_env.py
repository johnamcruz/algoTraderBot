#!/usr/bin/env python3
"""trail_exit_env.py — a PPO trailing-exit environment (strategy-agnostic).

Trains the *exit*, not the entry: instead of a fixed take-profit, a small PPO
agent manages a trailing stop bar-by-bar on the standard 0.5×ATR(20) stop every
strategy enters with. The agent only ever sees the trade's R-state (unrealized
R, MFE, stop distance, ATR/risk, time, momentum) — never how or why the trade
was entered — so the SAME policy applies to every strategy (supertrend, ema,
keltner, bos, …). SuperTrend flips are used only as a representative catalog of
NQ entry points to train on.

Pieces:
    build_arrays(df)      precompute close/high/low + trail ATR + stop ATR
    build_catalog(...)    every SuperTrend flip = one trainable "trade" sample
    TrailExitSim          pure-numpy simulator of a single trade (no gym dep)
    TrailingExitEnv       Gymnasium wrapper around the simulator (for SB3 PPO)
    NumpyMlpPolicy        torch-free loader for a trained policy (live inference)

The simulator is direction-agnostic: everything is framed in "favorable" units
(R-multiples) so a long and a short look identical to the agent.
"""
from __future__ import annotations

import numpy as np

# Reuse the exact indicators the live bot trades on.
import config
import indicators as ind
# ST_PERIOD/ST_MULT/ATR_P are structural (catalog + ATR periods) — bind at import.
# STOP_ATR / ACTIVATE_R / GIVEBACK_R are the TUNED exit knobs and are read from
# `config` at RUNTIME, so a per-timeframe exit config (config.apply_exit_config)
# drives the training sim exactly as it drives live (exit_manager reads them too).
from config import ST_PERIOD, ST_MULT, ATR_P

# ── trade / agent constants ────────────────────────────────────────────
MAX_HOLD = 80          # force-exit after this many bars (80 * 3min = 4h)
ATR_PERIOD = ST_PERIOD  # ATR used for the trail (same period as SuperTrend)
MOM_LOOKBACK = 3       # bars used for the momentum observation
OBS_DIM = 6
OBS_CLIP = 10.0

# The agent's action = pick a trailing-stop distance in ATR multiples.
# The stop is only ever ratcheted in the favorable direction.
TRAIL_MULTS = np.array([0.75, 1.0, 1.5, 2.0, 2.5, 3.5], dtype=np.float32)
N_ACTIONS = len(TRAIL_MULTS)


# ── data prep ──────────────────────────────────────────────────────────

def build_arrays(df, period: int = ST_PERIOD, mult: float = ST_MULT):
    """From an OHLC frame, precompute every per-bar array the sim needs.

    `atr` (period ATR_PERIOD) drives the trail distance; `atr_stop` (period
    ATR_P) sets the initial stop = STOP_ATR × atr_stop — the same 0.5×ATR(20)
    risk the live strategies enter with."""
    line, direction = ind.supertrend(df, period, mult)
    atr = np.asarray(ind.atr(df, ATR_PERIOD), dtype=np.float64)
    atr_stop = np.asarray(ind.atr(df, ATR_P), dtype=np.float64)
    return {
        "close": df["close"].to_numpy(dtype=np.float64),
        "high": df["high"].to_numpy(dtype=np.float64),
        "low": df["low"].to_numpy(dtype=np.float64),
        "atr": atr,
        "atr_stop": atr_stop,
        "line": np.asarray(line, dtype=np.float64),
        "direction": np.asarray(direction, dtype=np.float64),
    }


def build_catalog(arr, warmup: int = 150):
    """Every SuperTrend flip is a candidate trade. Returns an (N, 2) int array
    of (entry_idx, sign) where sign = +1 long / -1 short.

    Entry/stop/risk mirror the live bot: enter at the flip bar's close, initial
    stop = STOP_ATR × ATR(ATR_P) (the live 0.5×ATR risk). Flips with no valid
    ATR are dropped.
    """
    direction, atr, atr_stop = arr["direction"], arr["atr"], arr["atr_stop"]
    n = len(arr["close"])
    rows = []
    for i in range(warmup, n - 2):
        if direction[i] == direction[i - 1]:
            continue                                  # no flip
        if not (np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(atr_stop[i]) and atr_stop[i] > 0):
            continue
        sign = 1 if direction[i] == 1 else -1
        rows.append((i, sign))
    return np.asarray(rows, dtype=np.int64)


# ── single-trade simulator (pure numpy, no gym) ────────────────────────

class TrailExitSim:
    """Steps one trade forward. The agent sets the trail tightness each bar;
    the stop ratchets favorably and the trade ends on a stop hit, a max-hold
    timeout, or end-of-data. Reward each step = change in (un)realized R, so
    the discounted-undiscounted sum telescopes to the final realized R."""

    def __init__(self, arr):
        self.close = arr["close"]
        self.high = arr["high"]
        self.low = arr["low"]
        self.atr = arr["atr"]
        self.atr_stop = arr["atr_stop"]
        self.line = arr["line"]
        self.n = len(self.close)

    def reset(self, entry_idx: int, sign: int):
        self.sign = int(sign)
        self.entry_idx = int(entry_idx)
        self.i = int(entry_idx)                # last *observed* bar
        self.entry = float(self.close[entry_idx])
        # initial stop = STOP_ATR × ATR(ATR_P) — same 0.5×ATR(20) risk as live
        self.risk = config.STOP_ATR * float(self.atr_stop[entry_idx])
        self.stop = self.entry - self.sign * self.risk
        self.bars_held = 0
        self.mfe = 0.0           # close-based, for the observation (matches live)
        self.peak_R = 0.0        # bar-extreme peak, for the give-back cap
        self.prev_value = 0.0
        self.realized_R = None
        return self._obs()

    def _value_at(self, idx) -> float:
        return self.sign * (self.close[idx] - self.entry) / self.risk

    def _obs(self):
        i, s = self.i, self.sign
        cur = self.close[i]
        unreal = s * (cur - self.entry) / self.risk
        stop_dist = s * (cur - self.stop) / self.risk
        atr_R = self.atr[i] / self.risk
        bars_norm = self.bars_held / MAX_HOLD
        j = max(self.entry_idx, i - MOM_LOOKBACK)
        mom = s * (cur - self.close[j]) / self.risk
        # strategy-agnostic: purely the trade's R-state on the 0.5×ATR stop
        obs = np.array([unreal, self.mfe, stop_dist, atr_R,
                        bars_norm, mom], dtype=np.float32)
        return np.clip(obs, -OBS_CLIP, OBS_CLIP)

    def step(self, action: int):
        s = self.sign
        mult = float(TRAIL_MULTS[action])
        self.i += 1
        self.bars_held += 1
        done = False
        if self.i >= self.n:                          # ran out of history
            exit_price, done = self.close[self.i - 1], True
        else:
            i = self.i
            # TWO-TIER EXIT — models EXACTLY how the live bot fills, so the policy
            # trains on realistic prices (matches exit_manager.manage_trail +
            # SimBroker, the same design algoTraderAI uses: a resting broker stop
            # order kept at the floor, with a market close only as the backup).
            #
            #   prior_stop = the RESTING broker stop during this bar — the level
            #   set LAST bar (live reprices at each bar close, so intra-bar the
            #   resting order sits at last bar's floor).
            #
            #   • bar's unfavorable extreme crosses prior_stop → the resting stop
            #     order FILLS intra-bar AT that level (the give-back floor).
            #   • else it only crosses the floor we TIGHTEN to THIS bar (not yet a
            #     resting order) → live closes at MARKET → fill at the bar close
            #     (worse than the floor on a fast spike-and-reverse; this is the
            #     case the old sim modeled optimistically at the floor).
            prior_stop = self.stop
            fav = self.high[i] if s > 0 else self.low[i]
            self.peak_R = max(self.peak_R, s * (fav - self.entry) / self.risk)
            if self.peak_R >= config.ACTIVATE_R:
                cand = self.close[i] - s * mult * self.atr[i]
                cap = (self.entry + s * self.peak_R * self.risk) - s * config.GIVEBACK_R * self.risk
                cand = max(cand, cap) if s > 0 else min(cand, cap)
                new_stop = max(prior_stop, cand) if s > 0 else min(prior_stop, cand)
            else:
                new_stop = prior_stop
            unfav = self.low[i] if s > 0 else self.high[i]
            if (unfav <= prior_stop) if s > 0 else (unfav >= prior_stop):
                exit_price, done = prior_stop, True       # resting-stop fill (floor)
            elif (unfav <= new_stop) if s > 0 else (unfav >= new_stop):
                exit_price, done = self.close[i], True     # market-close backup (bar close)
            elif self.bars_held >= MAX_HOLD:
                exit_price, done = self.close[i], True      # max-hold timeout
            else:
                self.stop = new_stop                        # reprice resting stop for next bar
                exit_price = None

        if done:
            self.realized_R = float(s * (exit_price - self.entry) / self.risk)
            value = self.realized_R
            obs = self._obs() if self.i < self.n else self._terminal_obs()
        else:
            value = self._value_at(self.i)
            self.mfe = max(self.mfe, value)
            obs = self._obs()

        reward = value - self.prev_value
        self.prev_value = value
        return obs, float(reward), done, {"realized_R": self.realized_R}

    def _terminal_obs(self):
        self.i -= 1                                   # clamp for obs lookup
        obs = self._obs()
        self.i += 1
        return obs


# ── Gymnasium wrapper (training only) ──────────────────────────────────

def _make_gym_env(arr, catalog, seed=0):
    import gymnasium as gym
    from gymnasium import spaces

    class TrailingExitEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.sim = TrailExitSim(arr)
            self.catalog = catalog
            self.action_space = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.Box(
                -OBS_CLIP, OBS_CLIP, shape=(OBS_DIM,), dtype=np.float32)
            self._rng = np.random.default_rng(seed)

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            idx = self._rng.integers(len(self.catalog))
            entry_idx, sign = self.catalog[idx]
            return self.sim.reset(int(entry_idx), int(sign)), {}

        def step(self, action):
            obs, reward, done, info = self.sim.step(int(action))
            return obs, reward, done, False, info       # terminated, truncated

    return TrailingExitEnv()


def make_env(arr, catalog, seed=0):
    """Factory returning a fresh Gymnasium env (kept lazy so the numpy-only
    paths — simulator + live policy — never import gymnasium)."""
    return _make_gym_env(arr, catalog, seed)


# ── torch-free policy for live inference ───────────────────────────────

class NumpyMlpPolicy:
    """Loads a policy exported by train_ppo_exit.py (.npz of dense layers) and
    runs the forward pass in pure numpy — so the live bot can pick an action
    without importing torch/SB3 alongside xgboost (the OpenMP segfault trap)."""

    def __init__(self, layers):
        self.layers = layers                # list of (W, b); last is the head

    @classmethod
    def load(cls, path):
        data = np.load(path)
        n = int(data["n_layers"])
        layers = [(data[f"w{k}"], data[f"b{k}"]) for k in range(n)]
        return cls(layers)

    def action(self, obs) -> int:
        x = np.clip(np.asarray(obs, dtype=np.float32), -OBS_CLIP, OBS_CLIP)
        for W, b in self.layers[:-1]:
            x = np.tanh(W @ x + b)          # SB3 MlpPolicy default = Tanh
        W, b = self.layers[-1]
        logits = W @ x + b                  # action_net head
        return int(np.argmax(logits))

    def trail_mult(self, obs) -> float:
        return float(TRAIL_MULTS[self.action(obs)])
