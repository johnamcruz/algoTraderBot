"""Jev strategy (strategies/jev): when strategies fire, a reasoning model decides
which entry to take (or none) from every strategy's view, live signal and track
record.

Covers the swappable Decider interface (TypeSafe wire format, chat-model reply
parsing, caching, spec parsing), the causal track record, the state Jev sees,
and the strategy's entry decision. No network or model — HTTP and the
sub-strategies are faked."""
import json

import numpy as np
import pandas as pd
import pytest

import config
import strategies
from strategies.base import Signal
from strategies.jev import deciders as dm
from strategies.jev.deciders import (DeciderError, Decider, JevDecider,
                                     MlxDecider, OpenAIChatDecider, make_decider,
                                     parse_reply, system_prompt)
from strategies.jev.memory import JEV, TrackRecord
from strategies.jev.state import NONE, build_question, build_state
from strategies.jev.strategy import JevStrategy

OPTS = ["ema", "bos", NONE]


def _q(*names):
    """A question offering these strategies' (long) entries + none."""
    return build_question({n: Signal(n, 1, 1.0, 0.9, 0.1, 0, None) for n in names},
                          list(names))


def _bars(n=300, seed=0):
    rng = np.random.default_rng(seed)
    c = 20000 + np.cumsum(rng.normal(0, 5, n))
    return pd.DataFrame({
        "time": pd.date_range("2026-03-02 14:00", periods=n, freq="3min", tz="UTC"),
        "open": c, "high": c + 4, "low": c - 4, "close": c,
        "volume": rng.integers(100, 1000, n).astype(float),
    })


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


# ── deciders ────────────────────────────────────────────────────────────────

def test_jev_decider_posts_systemone_and_caches(monkeypatch, tmp_path):
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append((url, json, headers))
        return _Resp(200, {"answers": {"pick": {
            "type": "choice", "choice": "ema",
            "probabilities": {"ema": 0.8, "none": 0.2},
            "confidence": 0.7}}})

    monkeypatch.setattr(dm.requests, "post", fake_post)
    cache = str(tmp_path / "cache.jsonl")
    q = _q("ema")
    ans = JevDecider("jev-latest", api_key="k", cache_path=cache).decide({"x": 1}, q)

    assert ans["choice"] == "ema" and ans["probabilities"]["ema"] == 0.8
    url, body, headers = calls[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert headers["Authorization"] == "Bearer k"
    assert body == {"model": "jev-latest", "state": {"x": 1}, "questions": {"pick": q}}
    # a fresh decider with no key is answered from the disk cache
    assert JevDecider("jev-latest", api_key="", cache_path=cache).decide({"x": 1}, q) == ans
    assert len(calls) == 1


def test_jev_decider_errors(monkeypatch, tmp_path):
    c = str(tmp_path / "c.jsonl")
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


def test_parse_reply_takes_last_valid_json_after_thinking():
    text = ('<think>maybe {"choice": "bos"} ... no</think>\n'
            'Trend agrees across views.\n'
            '{"choice": "EMA", "confidence": 0.7, "reason": "ema+bos agree"}')
    a = parse_reply(text, OPTS)
    assert a["choice"] == "ema" and a["confidence"] == 0.7
    assert a["probabilities"] == {"ema": 0.7, "bos": 0.15, "none": 0.15}
    assert a["reason"] == "ema+bos agree"


def test_parse_reply_rejects_unknown_or_missing_choice():
    with pytest.raises(DeciderError):
        parse_reply('{"choice": "buy", "confidence": 1}', OPTS)
    with pytest.raises(DeciderError):
        parse_reply("<think>ema</think> I would take ema.", OPTS)


def test_openai_decider_sends_chat_and_parses(monkeypatch, tmp_path):
    sent = []

    def fake_post(url, json, headers, timeout):
        sent.append((url, json))
        return _Resp(200, {"choices": [{"message": {
            "content": '{"choice": "none", "confidence": 0.6, "reason": "chop"}'}}]})

    monkeypatch.setattr(dm.requests, "post", fake_post)
    d = OpenAIChatDecider("qwen3:4b", url="http://h:1/v1", api_key="",
                          cache_path=str(tmp_path / "c.jsonl"))
    q = _q("ema")
    assert d.decide({"s": 1}, q)["choice"] == "none"
    url, body = sent[0]
    assert url == "http://h:1/v1/chat/completions" and body["model"] == "qwen3:4b"
    assert body["messages"][0]["content"] == system_prompt(q)
    assert json.loads(body["messages"][1]["content"]) == {"s": 1}


def test_custom_decider_plugs_in_via_the_interface(tmp_path):
    class Always(Decider):
        backend = "always"

        def _decide(self, state, question):
            return {"choice": "none", "probabilities": {"none": 1.0},
                    "confidence": 1.0, "reason": ""}

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


# ── track record ────────────────────────────────────────────────────────────

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
    m.add("flat", _sig(+1), t0)
    m.update(_row("2026-01-01 10:00", 999, 0, 100))   # signal bar itself: ignored
    assert m.summary("win")["pending"] == 1
    m.update(_row("2026-01-01 10:03", 101.5, 99.5, 101))
    m.update(_row("2026-01-01 10:06", 102.0, 100.5, 100.5))
    assert m.summary("win")["last"] == "W"            # +2R on bar 2
    assert m.summary("loss")["last"] == "L"           # short stopped at 101 on bar 1
    f = m.summary("flat")
    assert f["last"] == "W"                           # also hit 102 on bar 2
    assert m.summary("win")["winRate"] == 1.0


def test_track_record_stop_first_on_straddle_and_timeout_marks_to_market():
    m = TrackRecord(target_r=2.0, max_bars=1, keep=10)
    t0 = pd.Timestamp("2026-01-01 10:00", tz="UTC")
    m.add("both", _sig(+1), t0)
    m.add("slow", _sig(+1, entry=200.0), t0)
    m.update(_row("2026-01-01 10:03", 105, 95, 100))  # straddles 'both's stop + target
    assert m.summary("both")["avgR"] == -1.0
    # 'slow' (entry 200, stop 199): bar low 95 → stopped too
    assert m.summary("slow")["last"] == "L"
    m.add("t", _sig(+1), pd.Timestamp("2026-01-01 10:03", tz="UTC"))
    m.update(_row("2026-01-01 10:06", 101, 99.5, 100.5))
    assert m.summary("t") == {"resolved": 1, "pending": 0, "winRate": 0.0,
                              "avgR": 0.5, "last": "F"}


def test_track_record_ignores_already_seen_bars():
    m = TrackRecord(target_r=2.0, max_bars=5, keep=10)
    m.add("a", _sig(+1), pd.Timestamp("2026-01-01 10:00", tz="UTC"))
    bar = _row("2026-01-01 10:03", 100.5, 99.5, 100)
    m.update(bar)
    m.update(bar)                                     # same bar again: no double count
    assert m.open[0]["bars"] == 1


# ── state / question ────────────────────────────────────────────────────────

def test_state_has_every_strategy_view_signal_and_record():
    bars = _bars()
    subs = [strategies.REGISTRY[n]() for n in
            ("supertrend", "ema", "keltner", "bos", "orb", "cisd_ote")]
    sig = Signal("ema", -1, 0.0, 0.0, 3.0, len(bars) - 1, None, proba=0.41, r_hat=0.5)
    mem = TrackRecord()
    st = build_state(bars, subs, {"ema": sig}, mem)
    text = json.dumps(st)                              # JSON-serializable

    assert list(st["strategies"]) == [s.name for s in subs]
    assert st["strategies"]["ema"]["signal"]["side"] == "short"
    assert st["strategies"]["ema"]["signal"]["modelWinProb"] == 0.41
    assert st["strategies"]["bos"]["signal"] is None
    assert st["strategies"]["supertrend"]["view"]["bias"] in ("up", "down")
    assert "gateOpen" in st["strategies"]["ema"]["view"]
    assert st["strategies"]["ema"]["trackRecord"] == {"resolved": 0, "pending": 0}
    assert st["yourRecentPicks"] == {"resolved": 0, "pending": 0}
    # no absolute prices or dates leak in (the model must not recognise the session)
    assert f"{bars['close'].iloc[-1]:.0f}" not in text
    assert "2026" not in text


def test_question_offers_fired_entries_plus_none_with_full_playbook():
    fired = {"ema": Signal("ema", 1, 1.0, 0.9, 0.1, 0, None),
             "bos": Signal("bos", -1, 1.0, 1.1, 0.1, 0, None)}
    q = build_question(fired, ["supertrend", "ema", "bos"])
    assert q["type"] == "choice"
    assert list(q["criteria"]) == ["ema", "bos", NONE]
    assert "long" in q["criteria"]["ema"] and "short" in q["criteria"]["bos"]
    assert list(q["instructions"]["playbook"]) == ["supertrend", "ema", "bos"]


# ── strategy ────────────────────────────────────────────────────────────────

class _Sub:
    """A fake sub-strategy that fires on chosen bar indices."""

    def __init__(self, name, fire_at=(), direction=1, proba=0.5):
        self.name, self.fire_at, self.d, self.proba = name, set(fire_at), direction, proba
        self.graded = 0

    def has_model(self):
        return True

    def detect(self, bars):
        i = len(bars) - 1
        if i not in self.fire_at and (i - len(bars)) not in self.fire_at:
            return None
        e = float(bars["close"].iloc[-1])
        return Signal(self.name, self.d, e, e - self.d * 2.0, 2.0, i,
                      bars["time"].iloc[-1])

    def grade(self, bars, sig, emb=None):
        self.graded += 1
        return self.proba, 1.0


class _Decider:
    spec = "fake:x"

    def __init__(self, choice="none", p=0.6, error=None):
        self.choice, self.p, self.error, self.calls = choice, p, error, []

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


def test_nothing_fired_means_no_decision():
    dec = _Decider("a")
    j = JevStrategy(subs=[_Sub("a"), _Sub("b")], decider=dec)
    assert _run(j, _bars()) is None
    assert dec.calls == []


def test_takes_the_entry_the_decider_picks_even_below_floor():
    config.PROBA_FLOOR = 0.9
    bars = _bars()
    a = _Sub("a", fire_at={-1}, direction=1, proba=0.7)
    b = _Sub("b", fire_at={-1}, direction=-1, proba=0.2)
    dec = _Decider("b", p=0.3)
    j = JevStrategy(subs=[a, b, _Sub("c")], decider=dec)
    sig = _run(j, bars)

    assert j.accepts(sig)                              # p=0.3 < floor, still taken
    assert sig.direction == -1 and sig.risk == 2.0     # b's own side and stop
    assert sig.stop == pytest.approx(sig.entry + 2.0)
    assert sig.strategy == j.name == "jev:b" and sig.proba == 0.3
    state, question = dec.calls[0]
    assert a.graded == b.graded == 1
    assert list(question["criteria"]) == ["a", "b", NONE]   # only fired entries
    assert list(state["strategies"]) == ["a", "b", "c"]     # but every strategy's data
    assert state["strategies"]["b"]["signal"]["modelWinProb"] == 0.2
    assert state["strategies"]["c"]["signal"] is None
    assert j.memory.summary(JEV)["pending"] == 1             # its pick is tracked


def test_none_failure_or_unoffered_pick_stays_flat():
    bars = _bars()
    for dec in (_Decider("none"), _Decider("zzz"),
                _Decider(error=DeciderError("down"))):
        j = JevStrategy(subs=[_Sub("a", fire_at={-1})], decider=dec)
        sig = _run(j, bars)
        assert not j.accepts(sig)
        assert j.name == "jev" and j.skip_reason == "jev: no pick"


def test_bootstrap_and_catch_up_fill_the_track_record():
    config.JEV_MEMORY_BOOT_BARS = 100
    bars = _bars(300)
    a = _Sub("a", fire_at={210, 250})                  # both inside the boot window
    j = JevStrategy(subs=[a], decider=_Decider("a"))
    _run(j, bars.iloc[:280])
    rec = j.memory.summary("a")
    assert rec["resolved"] + rec["pending"] == 2       # history replayed at startup

    # bars 280..289 pass while in a trade (no detect) — caught up on the next call
    a.fire_at.add(285)
    _run(j, bars.iloc[:290])
    rec = j.memory.summary("a")
    assert rec["resolved"] + rec["pending"] == 3


def test_registered_but_not_default():
    assert "jev" in strategies.REGISTRY
    assert "jev" not in config.ACTIVE_STRATEGIES
    subs = strategies.make_strategies(["jev"])[0].subs
    assert "jev" not in [s.name for s in subs]


def test_jev_strategies_config_limits_the_panel():
    config.JEV_STRATEGIES = ["ema", "bos"]
    assert [s.name for s in JevStrategy().subs] == ["ema", "bos"]


def test_other_strategies_keep_the_proba_floor():
    config.PROBA_FLOOR = 0.5
    s = strategies.EmaCrossStrategy()
    lo = Signal("ema", 1, 1.0, 0.9, 0.1, 0, None, proba=0.49)
    hi = Signal("ema", 1, 1.0, 0.9, 0.1, 0, None, proba=0.5)
    assert not s.accepts(lo) and s.accepts(hi)


def test_legacy_bundles_load_without_leaving_aliases():
    """The shipped bundles were pickled under pre-rename module paths; they load,
    and the temporary aliases are gone afterwards (never shadow other packages)."""
    import sys
    b = strategies.EmaCrossStrategy()._load_bundle()
    assert "signal_head" in b
    assert not [k for k in sys.modules
                if k.startswith(("pipelines", "futures_foundation.chronos"))]


# ── the full data feed: FFM context + each fired setup's model features ─────

def test_setup_feature_names_match_what_each_model_scores():
    """SETUP_FEATURES must line up 1:1 with each strategy's _hand_features tail —
    a mismatch would hand the decider mislabeled numbers."""
    from strategies import cisd_ote_detect as cod
    from strategies.base import FFM_COLS
    from strategies.jev.state import SETUP_FEATURES
    bars = _bars()
    i = len(bars) - 1
    for name in ("supertrend", "ema", "keltner", "bos", "orb"):
        tail = strategies.REGISTRY[name]()._hand_features(bars, i, 1)[len(FFM_COLS):]
        assert len(tail) == len(SETUP_FEATURES[name]), name
    assert len(SETUP_FEATURES["cisd_ote"]) == cod.N_GEOM
    assert set(SETUP_FEATURES) == set(strategies.REGISTRY) - {"jev"}


def test_state_carries_ffm_context_and_named_setup():
    bars = _bars()
    ema = strategies.EmaCrossStrategy()
    sig = Signal("ema", 1, 0.0, 0.0, 3.0, len(bars) - 1, None, proba=0.5)
    st = build_state(bars, [ema, strategies.BosStrategy()], {"ema": sig}, TrackRecord())
    ctx = st["context"]
    assert {"bar", "returns", "volume", "volatility", "session", "structure",
            "liquiditySweeps", "higherTimeframe", "time"} <= set(ctx)
    flat = {k for g in ctx.values() for k in g}
    assert len(flat) == 73                             # 76 minus the 3 dropped
    assert "vty_atr_raw" not in flat                   # raw price points never leak
    setup = st["strategies"]["ema"]["signal"]["setup"]
    assert list(setup) == ["ema_spread_atr", "slow_slope_atr", "price_vs_slow_atr",
                           "adx_div100", "adx_slope_div100"]
