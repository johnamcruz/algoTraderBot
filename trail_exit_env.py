#!/usr/bin/env python3
"""trail_exit_env.py — a PPO trailing-exit environment for the SuperTrend bot.

The entry decision stays exactly as it is in supertrend_ai_bot.py (SuperTrend
flip graded by the Chronos+XGBoost head). What this module replaces is the
*exit*: instead of a fixed 2R take-profit, a small PPO agent manages a trailing
stop bar-by-bar.

Pieces:
    build_arrays(df)      precompute close/high/low/atr + SuperTrend line/dir
    build_catalog(...)    every SuperTrend flip = one trainable "trade"
    TrailExitSim          pure-numpy simulator of a single trade (no gym dep)
    TrailingExitEnv       Gymnasium wrapper around the simulator (for SB3 PPO)
    NumpyMlpPolicy        torch-free loader for a trained policy (live inference)

The simulator is direction-agnostic: everything is framed in "favorable" units
(R-multiples) so a long and a short look identical to the agent.
"""
from __future__ import annotations

import numpy as np

# Reuse the exact indicators the live bot trades on.
from supertrend_ai_bot import _wilder_atr, supertrend, ST_PERIOD, ST_MULT

# ── trade / agent constants ────────────────────────────────────────────
MAX_HOLD = 80          # force-exit after this many bars (80 * 3min = 4h)
ATR_PERIOD = ST_PERIOD  # ATR used for the trail (same period as SuperTrend)
MOM_LOOKBACK = 3       # bars used for the momentum observation
OBS_DIM = 7
OBS_CLIP = 10.0

# The agent's action = pick a trailing-stop distance in ATR multiples.
# The stop is only ever ratcheted in the favorable direction.
TRAIL_MULTS = np.array([0.75, 1.0, 1.5, 2.0, 2.5, 3.5], dtype=np.float32)
N_ACTIONS = len(TRAIL_MULTS)


# ── data prep ──────────────────────────────────────────────────────────

def build_arrays(df, period: int = ST_PERIOD, mult: float = ST_MULT):
    """From an OHLC frame, precompute every per-bar array the sim needs."""
    line, direction = supertrend(df, period, mult)
    atr = _wilder_atr(df, ATR_PERIOD).to_numpy(dtype=np.float64)
    return {
        "close": df["close"].to_numpy(dtype=np.float64),
        "high": df["high"].to_numpy(dtype=np.float64),
        "low": df["low"].to_numpy(dtype=np.float64),
        "atr": atr,
        "line": np.asarray(line, dtype=np.float64),
        "direction": np.asarray(direction, dtype=np.float64),
    }


def build_catalog(arr, warmup: int = 150):
    """Every SuperTrend flip is a candidate trade. Returns an (N, 2) int array
    of (entry_idx, sign) where sign = +1 long / -1 short.

    Entry/stop/risk mirror the live bot: enter at the flip bar's close, initial
    stop at the SuperTrend line, risk = |entry - stop|. Flips with a
    wrong-side or zero-risk stop are dropped (the live bot skips them too).
    """
    close, line, direction, atr = (arr["close"], arr["line"],
                                   arr["direction"], arr["atr"])
    n = len(close)
    rows = []
    for i in range(warmup, n - 2):
        if direction[i] == direction[i - 1]:
            continue                                  # no flip
        sign = 1 if direction[i] == 1 else -1
        risk = (close[i] - line[i]) * sign            # entry - stop, favorable
        if risk <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
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
        self.line = arr["line"]
        self.n = len(self.close)

    def reset(self, entry_idx: int, sign: int):
        self.sign = int(sign)
        self.entry_idx = int(entry_idx)
        self.i = int(entry_idx)                # last *observed* bar
        self.entry = float(self.close[entry_idx])
        self.stop = float(self.line[entry_idx])   # initial protective stop
        self.risk = (self.entry - self.stop) * self.sign
        self.bars_held = 0
        self.mfe = 0.0
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
        dist_st = s * (cur - self.line[i]) / self.risk
        obs = np.array([unreal, self.mfe, stop_dist, atr_R,
                        bars_norm, mom, dist_st], dtype=np.float32)
        return np.clip(obs, -OBS_CLIP, OBS_CLIP)

    def step(self, action: int):
        s = self.sign
        mult = float(TRAIL_MULTS[action])
        # candidate trail off the just-closed bar, then ratchet favorably
        cand = self.close[self.i] - s * mult * self.atr[self.i]
        self.stop = max(self.stop, cand) if s > 0 else min(self.stop, cand)

        self.i += 1
        self.bars_held += 1
        done = False
        if self.i >= self.n:                          # ran out of history
            exit_price, done = self.close[self.i - 1], True
        else:
            hi, lo = self.high[self.i], self.low[self.i]
            hit = (lo <= self.stop) if s > 0 else (hi >= self.stop)
            if hit:
                exit_price, done = self.stop, True    # filled at the stop
            elif self.bars_held >= MAX_HOLD:
                exit_price, done = self.close[self.i], True
            else:
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
