"""Per-timeframe exit config (exit_configs.json). The best ACTIVATE_R/GIVEBACK_R/
STOP_ATR differ by timeframe, so they're stored per-minute and applied for the
active timeframe — read at runtime by both the live exit and the training sim."""
import json

import config
from ppo_exit import optimize_exit as oe


def _write(tmp_path, data):
    p = tmp_path / "exit_configs.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_apply_sets_globals_for_timeframe(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXIT_CONFIGS_PATH",
                        _write(tmp_path, {"1": {"ACTIVATE_R": 1.5, "GIVEBACK_R": 1.0,
                                                "STOP_ATR": 0.9}}))
    out = config.apply_exit_config(1)
    assert out == {"ACTIVATE_R": 1.5, "GIVEBACK_R": 1.0, "STOP_ATR": 0.9}
    assert config.ACTIVATE_R == 1.5 and config.GIVEBACK_R == 1.0 and config.STOP_ATR == 0.9


def test_missing_timeframe_key_keeps_defaults(tmp_path, monkeypatch):
    config.ACTIVATE_R, config.GIVEBACK_R, config.STOP_ATR = 2.0, 0.75, 0.5
    monkeypatch.setattr(config, "EXIT_CONFIGS_PATH",
                        _write(tmp_path, {"3": {"ACTIVATE_R": 9.9}}))
    assert config.apply_exit_config(1) is None          # no "1" key
    assert (config.ACTIVATE_R, config.GIVEBACK_R, config.STOP_ATR) == (2.0, 0.75, 0.5)


def test_missing_file_is_noop(tmp_path, monkeypatch):
    config.ACTIVATE_R = 2.0
    monkeypatch.setattr(config, "EXIT_CONFIGS_PATH", str(tmp_path / "nope.json"))
    assert config.apply_exit_config(1) is None
    assert config.ACTIVATE_R == 2.0


def test_different_timeframes_get_different_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXIT_CONFIGS_PATH",
                        _write(tmp_path, {"1": {"ACTIVATE_R": 1.5, "GIVEBACK_R": 0.75, "STOP_ATR": 1.0},
                                          "3": {"ACTIVATE_R": 2.5, "GIVEBACK_R": 0.75, "STOP_ATR": 0.7}}))
    assert config.apply_exit_config(1)["STOP_ATR"] == 1.0
    assert config.apply_exit_config(3)["STOP_ATR"] == 0.7


def test_save_config_roundtrips_and_preserves_keys(tmp_path, monkeypatch):
    path = _write(tmp_path, {"_note": "keep me", "3": {"ACTIVATE_R": 2.0,
                                                       "GIVEBACK_R": 0.75, "STOP_ATR": 0.5}})
    monkeypatch.setattr(config, "EXIT_CONFIGS_PATH", path)
    oe._save_config(1, {"ACTIVATE_R": 1.5, "GIVEBACK_R": 0.5, "STOP_ATR": 0.9})
    saved = json.loads(open(path).read())
    assert saved["1"] == {"ACTIVATE_R": 1.5, "GIVEBACK_R": 0.5, "STOP_ATR": 0.9}
    assert saved["3"]["STOP_ATR"] == 0.5 and saved["_note"] == "keep me"   # untouched
