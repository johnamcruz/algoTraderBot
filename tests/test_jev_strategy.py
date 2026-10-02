"""Jev strategy (strategies/jev): a reasoning model IS the strategy — each flat bar
it decides long / short / none itself, from all the context the six strategies
have (views, signals, track records), the consolidated market context and its own
last decisions. The six strategies never trade under jev.

Covers the swappable Decider interface (wire format, validation, retries, cache
key, reply parsing, startup checks), the causal track record + decision log, the
consolidated state (no duplicates, readable, no leaks, in-sample probabilities
withheld), the question, and the strategy's decision/entry rules — plus the bot
fixes (fill rebase, jev must run alone). No network or model — HTTP and the
sub-strategies are faked."""
import json

import numpy as np
import pandas as pd
import pytest

import bot
import config
import strategies
from strategies.base import Signal
from strategies.jev import deciders as dm
from strategies.jev.deciders import (DeciderError, Decider, JevDecider,
                                     MlxDecider, OpenAIChatDecider, make_decider,
                                     parse_reply, system_prompt, validate)
from strategies.jev.memory import JEV, DecisionLog, TrackRecord
from strategies.jev.state import (LONG, NONE, SHORT, SETUP_FEATURES,
                                  build_question, build_state, market_context)
from strategies.jev.strategy import JevStrategy

OPTS = [LONG, SHORT, NONE]


def _bars(n=300, seed=0, start="2026-03-02 14:00"):
    rng = np.random.default_rng(seed)
    c = 20000 + np.cumsum(rng.normal(0, 5, n))
    return pd.DataFrame({
        "time": pd.date_range(start, periods=n, freq="3min", tz="UTC"),
        "open": c, "high": c + 4, "low": c - 4, "close": c,
        "volume": rng.integers(100, 1000, n).astype(float),
    })


def _q():
    return build_question(["ema", "bos"])


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _jev_reply(choice="long", probs=None, model="jev-2026-09"):
    return {"model": model, "answers": {"pick": {
        "type": "choice", "choice": choice,
        "probabilities": probs or {"long": 0.6, "short": 0.1, "none": 0.3},
        "confidence": 0.4}}}


# ── deciders ────────────────────────────────────────────────────────────────

def test_jev_decider_posts_systemone_validates_and_caches(monkeypatch, tmp_path):
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append((url, json, headers, timeout))
        return _Resp(200, _jev_reply())

    monkeypatch.setattr(dm.requests, "post", fake_post)
    cache = str(tmp_path / "cache.jsonl")
    q = _q()
    ans = JevDecider("jev-latest", api_key="k", cache_path=cache).decide({"x": 1}, q)

    assert ans["choice"] == "long" and ans["probabilities"]["long"] == 0.6
    assert ans["model"] == "jev-2026-09"               # resolved version is kept
    url, body, headers, timeout = calls[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert headers["Authorization"] == "Bearer k"
    assert timeout == config.JEV_API_TIMEOUT           # short — Jev answers in ~100 ms
    assert body == {"model": "jev-latest", "state": {"x": 1}, "questions": {"pick": q}}
    # a fresh decider with no key is answered from the disk cache
    assert JevDecider("jev-latest", api_key="", cache_path=cache).decide({"x": 1}, q) == ans
    assert len(calls) == 1


def test_jev_decider_retries_rate_limits(monkeypatch, tmp_path):
    replies = [_Resp(429, {"m": "slow down"}), _Resp(529, {"m": "busy"}),
               _Resp(200, _jev_reply("short"))]
    monkeypatch.setattr(dm.requests, "post", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(dm.time, "sleep", lambda s: None)
    d = JevDecider(api_key="k", cache_path=str(tmp_path / "c.jsonl"))
    assert d.decide({}, _q())["choice"] == "short"
    assert replies == []


def test_jev_decider_errors_become_decider_errors(monkeypatch, tmp_path):
    c = str(tmp_path / "c.jsonl")
    monkeypatch.setattr(dm.time, "sleep", lambda s: None)
    with pytest.raises(DeciderError, match="TYPESAFE_AI_API_KEY"):
        JevDecider(api_key="", cache_path=c).decide({}, _q())
    monkeypatch.setattr(dm.requests, "post", lambda *a, **k: _Resp(401, {"m": "bad"}))
    with pytest.raises(DeciderError, match="401"):
        JevDecider(api_key="k", cache_path=c).decide({}, _q())

    def boom(*a, **k):
        raise dm.requests.Timeout("slow")
    monkeypatch.setattr(dm.requests, "post", boom)
    with pytest.raises(DeciderError, match="request failed"):
        JevDecider(api_key="k", cache_path=c).decide({}, _q())


def test_malformed_answers_are_rejected_and_never_cached(monkeypatch, tmp_path):
    cache = tmp_path / "c.jsonl"
    for bad in ({"answers": {"pick": {"choice": ["long"]}}},          # not a string
                {"answers": {"pick": {"choice": "buy"}}},             # not offered
                {"answers": {"pick": "long"}}):                       # wrong shape
        monkeypatch.setattr(dm.requests, "post", lambda *a, b=bad, **k: _Resp(200, b))
        with pytest.raises(DeciderError):
            JevDecider(api_key="k", cache_path=str(cache)).decide({}, _q())
    assert not cache.exists()


def test_validate_normalizes_choice_and_clamps_probabilities():
    a = validate({"choice": " LONG ", "probabilities": {"long": 1.7, "Short": -2,
                                                       "none": float("nan"), "x": "?"},
                  "confidence": 3}, OPTS)
    assert a["choice"] == "long"
    assert a["probabilities"] == {"long": 1.0, "short": 0.0}
    assert a["confidence"] == 1.0


def test_backend_crash_is_a_decider_error(tmp_path):
    class Crashy(Decider):
        backend = "crashy"

        def _decide(self, state, question):
            raise KeyError("boom")

    with pytest.raises(DeciderError, match="KeyError"):
        Crashy("x", cache_path=str(tmp_path / "c.jsonl")).decide({}, _q())


def test_parse_reply_handles_thinking_nested_json_and_braces():
    text = ('<think>maybe {"choice": "short"} ... no</think>\n'
            'Trend agrees across views {strongly}.\n'
            '{"choice": "Long", "confidence": 0.7, "reason": "ADX {25} rising", '
            '"probabilities": {"long": 0.7, "none": 0.3}}')
    a = parse_reply(text, OPTS)
    assert a["choice"] == "long" and a["confidence"] == 0.7
    assert a["probabilities"] == {"long": 0.7, "short": 0.15, "none": 0.15}
    assert a["reason"] == "ADX {25} rising"


def test_parse_reply_rejects_unknown_or_missing_choice():
    with pytest.raises(DeciderError):
        parse_reply('{"choice": "buy", "confidence": 1}', OPTS)
    with pytest.raises(DeciderError):
        parse_reply("<think>long</think> I would go long.", OPTS)


def test_openai_decider_sends_chat_and_cache_key_tracks_settings(monkeypatch,
                                                                 tmp_path):
    sent = []

    def fake_post(url, json, headers, timeout):
        sent.append((url, json))
        return _Resp(200, {"choices": [{"message": {
            "content": '{"choice": "none", "confidence": 0.6, "reason": "chop"}'}}]})

    monkeypatch.setattr(dm.requests, "post", fake_post)
    d = OpenAIChatDecider("qwen3:4b", url="http://h:1/v1", api_key="",
                          cache_path=str(tmp_path / "c.jsonl"))
    q = _q()
    assert d.decide({"s": 1}, q)["choice"] == "none"
    url, body = sent[0]
    assert url == "http://h:1/v1/chat/completions" and body["model"] == "qwen3:4b"
    assert body["messages"][0]["content"] == system_prompt(q)
    assert json.loads(body["messages"][1]["content"]) == {"s": 1}

    d.decide({"s": 1}, q)                               # same request → cached
    assert len(sent) == 1
    config.JEV_LLM_THINK = not config.JEV_LLM_THINK     # a setting changed → re-ask
    d.decide({"s": 1}, q)
    assert len(sent) == 2


def test_custom_decider_plugs_in_via_the_interface(tmp_path):
    class Always(Decider):
        backend = "always"

        def _decide(self, state, question):
            return {"choice": "none", "probabilities": {"none": 1.0}}

    d = Always("x", cache_path=str(tmp_path / "c.jsonl"))
    assert d.spec == "always:x"
    assert d.decide({}, _q())["choice"] == "none"


def test_make_decider_parses_backend_and_model():
    assert isinstance(make_decider("jev:jev-latest"), JevDecider)
    m = make_decider("mlx:mlx-community/Qwen3-0.6B-4bit")
    assert isinstance(m, MlxDecider) and m.model == "mlx-community/Qwen3-0.6B-4bit"
    o = make_decider("openai:qwen3:4b")             # model may contain colons
    assert isinstance(o, OpenAIChatDecider) and o.model == "qwen3:4b"
    config.JEV_DECIDER = "mlx:some/model"
    assert make_decider().spec == "mlx:some/model"
    with pytest.raises(SystemExit):
        make_decider("nope:x")


def test_startup_check_fails_fast_on_missing_key():
    with pytest.raises(SystemExit, match="TYPESAFE_AI_API_KEY"):
        JevDecider(api_key="").check()


# ── track record + decision log ─────────────────────────────────────────────

def _row(t, high, low, close):
    return pd.Series({"time": pd.Timestamp(t, tz="UTC"), "high": high,
                      "low": low, "close": close})


def _sig(d, entry=100.0, risk=1.0):
    return Signal("x", d, entry, entry - d * risk, risk, 0, None)


def test_track_record_scores_target_stop_and_timeout():
    m = TrackRecord(target_r=2.0, max_bars=2, keep=10)
    t0 = pd.Timestamp("2026-01-01 10:00", tz="UTC")
    m.add("win", _sig(+1), t0)
    m.add("loss", _sig(-1), t0)
    m.update(_row("2026-01-01 10:00", 999, 0, 100))   # its own bar: never scored
    assert m.summary("win")["pending"] == 1
    m.update(_row("2026-01-01 10:03", 101.5, 99.5, 101))
    m.update(_row("2026-01-01 10:06", 102.0, 100.5, 100.5))
    assert m.summary("win")["last"] == "W"            # +2R on bar 2
    assert m.summary("loss")["last"] == "L"           # short stopped at 101 on bar 1
    m.add("t", _sig(+1), pd.Timestamp("2026-01-01 10:06", tz="UTC"))
    m.update(_row("2026-01-01 10:09", 100.5, 99.5, 100.5))
    m.update(_row("2026-01-01 10:12", 101, 99.5, 100.5))
    assert m.summary("t") == {"resolved": 1, "pending": 0, "winRate": 0.0,
                              "avgR": 0.5, "last": "F"}


def test_track_record_stop_first_on_straddle_and_ignores_seen_bars():
    m = TrackRecord(target_r=2.0, max_bars=5, keep=10)
    m.add("both", _sig(+1), pd.Timestamp("2026-01-01 10:00", tz="UTC"))
    m.update(_row("2026-01-01 10:03", 105, 95, 100))
    assert m.summary("both")["avgR"] == -1.0
    m.add("a", _sig(+1), pd.Timestamp("2026-01-01 10:03", tz="UTC"))
    quiet = _row("2026-01-01 10:06", 100.5, 99.5, 100)
    m.update(quiet)
    m.update(quiet)                                   # same bar again: no double count
    assert m.open[0]["bars"] == 1


def test_track_record_timeout_scales_with_stop_width():
    config.STOP_ATR = 0.5
    m = TrackRecord(target_r=2.0, max_bars=20, keep=10)
    t0 = pd.Timestamp("2026-01-01 10:00", tz="UTC")
    std = m.add("std", _sig(+1, risk=5.0), t0, atr=10.0)      # 0.5 ATR stop
    wide = m.add("wide", _sig(+1, risk=20.0), t0, atr=10.0)   # 2 ATR stop
    assert std["max_bars"] == 20 and wide["max_bars"] == 80


def test_decision_log_shows_both_sides_and_keeps_the_last_n():
    log_ = DecisionLog(keep=2, target_r=2.0)
    t = pd.Timestamp("2026-01-01 10:00", tz="UTC")
    for k in range(3):
        log_.add(t + pd.Timedelta(minutes=3 * k), {"adx": 20}, ["bos long"],
                 {"long": {"r": 2.0}, "short": {"r": -1.0}}, NONE if k else LONG)
    out = log_.render(t + pd.Timedelta(minutes=30))
    assert len(out) == 2                              # only the last `keep`
    assert out[-1] == {"minutesAgo": 24, "situation": {"adx": 20},
                       "signalsFired": ["bos long"], "yourChoice": "none",
                       "longOutcome": "W +2.0R", "shortOutcome": "L -1.0R",
                       "result": "stayed flat; long would have been W +2.0R, "
                                 "short would have been L -1.0R"}
    log_.add(t, {}, [], {"long": None, "short": {}}, LONG)
    assert log_.render(t)[-1]["result"] == "took long: pending"


# ── state: consolidated, readable, causal ───────────────────────────────────

def _panel():
    return [strategies.REGISTRY[n]() for n in
            ("supertrend", "ema", "keltner", "bos", "orb", "cisd_ote")]


def test_state_has_market_context_once_and_every_strategy():
    bars = _bars()
    sig = Signal("bos", -1, 0.0, 0.0, 3.0, len(bars) - 1, None, proba=0.41, r_hat=0.5)
    st = build_state(bars, _panel(), {"bos": sig}, TrackRecord())
    text = json.dumps(st, allow_nan=False)             # valid JSON, no NaN
    m = st["market"]
    assert set(m) == {"time", "trend", "momentum", "volatility", "volume", "lastBar",
                      "session", "range", "liquiditySweeps"}
    assert list(st["strategies"]) == ["supertrend", "ema", "keltner", "bos", "orb",
                                      "cisd_ote"]
    assert st["strategies"]["supertrend"]["view"]["bias"] in ("up", "down")
    bos = st["strategies"]["bos"]["signal"]
    assert bos["side"] == "short" and list(bos["setup"]) == ["breakStrengthAtr"]
    assert st["strategies"]["ema"]["signal"] is None
    # readable labels, no raw FFM codes or names
    assert m["lastBar"]["type"] in ("doji", "bull strength", "bear strength",
                                    "bull pin bar", "bear pin bar", "mixed", None)
    assert isinstance(m["liquiditySweeps"]["1h"]["bullishActive"], bool)
    for raw in ("candle_type", "sess_id", "vol_delta_proxy", "vty_atr_raw",
                "tmp_hour", "ret_close_1", "str_swing_high_dist"):
        assert raw not in text
    # ADX lives once, in market.trend — not repeated in views or setups
    assert text.count('"adx"') == 1
    # no absolute prices or dates leak in
    assert f"{bars['close'].iloc[-1]:.0f}" not in text
    assert "2026" not in text


def test_in_sample_model_probability_is_withheld():
    """A model's win probability on a bar inside its training span is a leaked
    label — the state must withhold it."""
    bars = _bars()
    ema = strategies.EmaCrossStrategy()
    sig = Signal("ema", 1, 0.0, 0.0, 3.0, len(bars) - 1, bars["time"].iloc[-1],
                 proba=0.9, r_hat=2.0)
    ema.train_end = lambda: bars["time"].iloc[-1] + pd.Timedelta(days=1)
    s_in = build_state(bars, [ema], {"ema": sig}, TrackRecord())["strategies"]["ema"]
    assert s_in["signal"]["modelWinProb"] is None
    assert s_in["signal"]["modelExpectedR"] is None
    assert s_in["signal"]["modelInSample"] is True
    ema.train_end = lambda: bars["time"].iloc[-1] - pd.Timedelta(days=1)
    s_out = build_state(bars, [ema], {"ema": sig}, TrackRecord())["strategies"]["ema"]
    assert s_out["signal"]["modelWinProb"] == 0.9
    assert "modelInSample" not in s_out["signal"]


def test_market_context_has_no_lookahead():
    """The market context at bar i must be identical whether or not later bars
    exist in the frame."""
    bars = _bars(400)
    for i in (250, 320, 349):
        assert market_context(bars, i) == market_context(bars.iloc[:i + 1], i)


def test_setup_feature_names_match_what_each_model_scores():
    from strategies import cisd_ote_detect as cod
    from strategies.base import FFM_COLS
    bars = _bars()
    i = len(bars) - 1
    for name in ("supertrend", "ema", "keltner", "bos", "orb"):
        tail = strategies.REGISTRY[name]()._hand_features(bars, i, 1)[len(FFM_COLS):]
        assert len(tail) == len(SETUP_FEATURES[name]), name
    assert len(SETUP_FEATURES["cisd_ote"]) == cod.N_GEOM
    assert set(SETUP_FEATURES) == set(strategies.REGISTRY) - {"jev"}


def test_question_is_long_short_none_with_breakeven_timing_and_playbook():
    config.JEV_MEMORY_TARGET_R = 2.0
    q = build_question(["ema", "bos"])
    assert q["type"] == "choice"
    assert list(q["criteria"]) == [LONG, SHORT, NONE]
    ins = q["instructions"]
    assert "0.33" in ins["goal"]                       # 2:1 → break-even ≈ 1/3
    assert "timing" in ins and "recentDecisions" in ins["inputs"]
    assert list(ins["playbook"]) == ["ema", "bos"]


# ── strategy: Jev decides long / short / none itself ────────────────────────

class _Sub:
    """A fake strategy that fires on chosen bar indices (-1 = the last bar)."""

    def __init__(self, name, fire_at=(), direction=1, proba=0.5, risk=2.0):
        self.name, self.fire_at, self.d, self.proba = name, set(fire_at), direction, proba
        self.risk = risk
        self.graded = 0

    def has_model(self):
        return True

    def detect(self, bars):
        i = len(bars) - 1
        if i not in self.fire_at and -1 not in self.fire_at:
            return None
        e = float(bars["close"].iloc[-1])
        return Signal(self.name, self.d, e, e - self.d * self.risk, self.risk, i,
                      bars["time"].iloc[-1])

    def grade(self, bars, sig, emb=None):
        self.graded += 1
        return self.proba, 1.0


class _Decider:
    spec = "fake:x"

    def __init__(self, choice="none", p=0.6, error=None):
        self.choice, self.p, self.error, self.calls = choice, p, error, []
        self.checked = False

    def check(self):
        self.checked = True

    def decide(self, state, question):
        self.calls.append((state, question))
        if self.error:
            raise self.error
        return {"choice": self.choice, "probabilities": {self.choice: self.p},
                "confidence": self.p, "reason": "test"}


EMB = np.zeros((1, 256))


def _run(j, bars):
    sig = j.detect(bars)
    if sig is not None:
        j.grade(bars, sig, emb=EMB)
    return sig


def test_decides_on_every_new_bar_even_without_signals():
    dec = _Decider(LONG, p=0.3)
    j = JevStrategy(subs=[_Sub("a")], decider=dec)
    config.PROBA_FLOOR = 0.9
    sig = _run(j, _bars())
    assert len(dec.calls) == 1                         # asked with nothing firing
    assert j.accepts(sig)                              # p=0.3 < floor: still enters
    assert sig.direction == 1 and sig.strategy == j.name == "jev:long"
    assert sig.risk == pytest.approx(sig.entry - sig.stop)


def test_only_on_signals_mode_skips_quiet_bars():
    config.JEV_ONLY_ON_SIGNALS = True
    dec = _Decider(LONG)
    j = JevStrategy(subs=[_Sub("a")], decider=dec)
    assert _run(j, _bars()) is None and dec.calls == []


def test_short_uses_the_standard_stop_not_a_strategy_stop():
    config.STOP_ATR = 0.5
    bars = _bars()
    j = JevStrategy(subs=[_Sub("a", fire_at={-1}, direction=-1, risk=50.0)],
                    decider=_Decider(SHORT, p=0.7))
    sig = _run(j, bars)
    assert sig.direction == -1 and sig.stop == pytest.approx(sig.entry + sig.risk)
    from indicators import atr
    assert sig.risk == pytest.approx(0.5 * atr(bars, config.ATR_P)[-1])
    assert sig.proba == 0.7


def test_none_or_failure_stays_flat():
    for dec in (_Decider(NONE), _Decider(error=DeciderError("down"))):
        j = JevStrategy(subs=[_Sub("a")], decider=dec)
        sig = _run(j, _bars())
        assert not j.accepts(sig)
        assert j.name == "jev" and j.skip_reason == "jev: none"


def test_a_bar_is_never_decided_twice():
    """Live, the broker can return the same last bar again (late publish, halt,
    weekend) — re-asking could flip a `none` into a late entry."""
    bars = _bars()
    dec = _Decider(NONE)
    j = JevStrategy(subs=[_Sub("a", fire_at={-1})], decider=dec)
    _run(j, bars)
    assert _run(j, bars) is None
    assert len(dec.calls) == 1 and len(j.history.entries) == 1


def test_slow_live_decision_skips_the_entry(monkeypatch):
    import strategies.jev.strategy as js
    config.LIVE = True
    config.JEV_MAX_DECISION_SEC = 20.0
    clock = iter([0.0, 25.0])                           # detect at 0s, decided at 25s
    monkeypatch.setattr(js.time, "monotonic", lambda: next(clock))
    j = JevStrategy(subs=[_Sub("a")], decider=_Decider(LONG))
    sig = _run(j, _bars())
    assert not j.accepts(sig)


def test_all_context_reaches_the_decider():
    a = _Sub("a", fire_at={-1}, direction=-1, proba=0.62)
    b = _Sub("b")
    dec = _Decider(SHORT)
    j = JevStrategy(subs=[a, b], decider=dec)
    _run(j, _bars())
    state, question = dec.calls[0]
    assert a.graded == 1 and b.graded == 0
    assert state["strategies"]["a"]["signal"]["side"] == "short"
    assert state["strategies"]["a"]["signal"]["modelWinProb"] == 0.62
    assert state["strategies"]["b"]["signal"] is None
    assert {"market", "strategies", "yourRecentPicks", "recentDecisions"} <= set(state)
    assert list(question["criteria"]) == [LONG, SHORT, NONE]


def test_recent_decisions_are_fed_back_with_causal_outcomes():
    bars = _bars(300)
    dec = _Decider(NONE)
    j = JevStrategy(subs=[_Sub("a", fire_at={-1})], decider=dec)

    _run(j, bars.iloc[:250])                           # decision 1
    assert dec.calls[0][0]["recentDecisions"] == []
    _run(j, bars.iloc[:251])                           # decision 2, one bar later
    d = dec.calls[1][0]["recentDecisions"][0]
    assert d["minutesAgo"] == 3 and d["yourChoice"] == "none"
    assert d["signalsFired"] == ["a long"]
    # resolved only from bars that have closed since — one bar of a 0.5 ATR stop
    # may or may not have been hit, but never from the future
    assert d["longOutcome"] == "pending" or d["longOutcome"][0] in "WLF"
    assert {"session", "adx", "last20BarsAtr"} <= set(d["situation"])
    assert "2026" not in json.dumps(dec.calls[1][0]["recentDecisions"])


def test_decision_outcomes_stay_pending_until_resolved():
    bars = _bars(300)
    j = JevStrategy(subs=[_Sub("a", fire_at={-1})], decider=_Decider(NONE))
    _run(j, bars.iloc[:250])
    e = j.history.entries[0]
    assert "r" not in e["sides"]["long"]               # nothing has closed yet
    assert j.history.render(bars["time"].iloc[249])[0]["longOutcome"] == "pending"


def test_quiet_flat_bars_are_not_logged_as_examples():
    bars = _bars(300)
    j = JevStrategy(subs=[_Sub("a")], decider=_Decider(NONE))
    for k in range(250, 255):
        _run(j, bars.iloc[:k])
    assert len(j.history.entries) == 0                 # no signal, no entry
    j._decider = _Decider(LONG)
    _run(j, bars.iloc[:256])
    assert len(j.history.entries) == 1
    rec = j.memory.summary(JEV)
    assert rec["pending"] + rec["resolved"] == 1


def test_bootstrap_and_catch_up_fill_the_track_record():
    config.JEV_MEMORY_BOOT_BARS = 100
    bars = _bars(300)
    a = _Sub("a", fire_at={210, 250})                  # both inside the boot window
    j = JevStrategy(subs=[a], decider=_Decider(NONE))
    _run(j, bars.iloc[:280])
    rec = j.memory.summary("a")
    assert rec["resolved"] + rec["pending"] == 2       # history replayed at startup
    a.fire_at.add(285)                                 # fires while "in a trade"
    _run(j, bars.iloc[:290])
    rec = j.memory.summary("a")
    assert rec["resolved"] + rec["pending"] == 3


def test_prepare_checks_the_decider_and_reset_clears_memory():
    dec = _Decider(LONG)
    j = JevStrategy(subs=[_Sub("a")], decider=dec)
    j.prepare()
    assert dec.checked
    _run(j, _bars())
    j.reset()                                          # contract roll
    assert j.memory.summary(JEV) == {"resolved": 0, "pending": 0}
    assert len(j.history.entries) == 0 and j._last_bar is None


def test_registered_alone_and_not_default():
    assert "jev" in strategies.REGISTRY
    assert "jev" not in config.ACTIVE_STRATEGIES
    subs = strategies.make_strategies(["jev"])[0].subs
    assert "jev" not in [s.name for s in subs]
    with pytest.raises(SystemExit, match="on its own"):
        strategies.make_strategies(["jev", "ema"])


def test_jev_strategies_config_limits_and_validates_the_panel():
    config.JEV_STRATEGIES = ["ema", "bos", "jev"]       # self-reference is ignored
    assert [s.name for s in JevStrategy().subs] == ["ema", "bos"]
    config.JEV_STRATEGIES = ["ema", "nope"]
    with pytest.raises(SystemExit, match="unknown"):
        JevStrategy().subs


def test_other_strategies_keep_the_proba_floor():
    config.PROBA_FLOOR = 0.5
    s = strategies.EmaCrossStrategy()
    lo = Signal("ema", 1, 1.0, 0.9, 0.1, 0, None, proba=0.49)
    hi = Signal("ema", 1, 1.0, 0.9, 0.1, 0, None, proba=0.5)
    assert not s.accepts(lo) and s.accepts(hi)


def test_legacy_bundles_load_without_leaving_aliases():
    import sys
    b = strategies.EmaCrossStrategy()._load_bundle()
    assert "signal_head" in b
    assert not [k for k in sys.modules
                if k.startswith(("pipelines", "futures_foundation.chronos"))]


# ── bot: the exit's R accounting is anchored to the actual fill ─────────────

def test_trade_state_is_rebased_on_the_fill(monkeypatch):
    class Client:
        pos = None

        def open_position(self, *a):
            return self.pos

        def cancel_orders(self, *a):
            return 0

        def place_market_with_stop(self, *a, **k):
            self.pos = {"averagePrice": 103.0}         # filled 3 points worse

    class Strat:
        name = "s"
        skip_reason = ""

        def detect(self, bars):
            return Signal("s", 1, 100.0, 98.0, 2.0, len(bars) - 1, bars["time"].iloc[-1])

        def grade(self, bars, sig, emb=None):
            return 0.9, 1.0

        def accepts(self, sig):
            return True

    ctx = type("Ctx", (), {})()
    ctx.client, ctx.account_id, ctx.contract_id = Client(), 0, "X"
    ctx.tick_size, ctx.tick_value, ctx.log_candles = 0.25, 5.0, False
    ctx.strategies, ctx.policy, ctx.trailing = [Strat()], object(), False
    monkeypatch.setattr(bot.strat, "embed_context", lambda bars, i: EMB)
    st = bot.handle_bar(ctx, _bars(), None)
    assert st["entry"] == 103.0 and st["stop"] == 101.0 and st["risk"] == 2.0
