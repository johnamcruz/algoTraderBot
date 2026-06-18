"""Micro → parent symbol mapping. Micros (MNQ/MES/…) trade their parent's bars and
are graded with the parent instrument's features, so base_symbol() must map every
micro to its parent and pass full-size / unknown symbols through unchanged. A wrong
mapping silently feeds the model the wrong instrument."""
import config


def test_every_micro_maps_to_its_parent():
    for micro, parent in config.MICRO_PARENT.items():
        assert config.base_symbol(micro) == parent
        assert micro.startswith("M")                    # micros are the M-prefixed contracts


def test_trained_micros_map_to_trained_parents():
    # the micros whose parent the bot actually has a model for
    for micro in ("MNQ", "MES", "M2K", "MYM", "MGC"):
        assert config.base_symbol(micro) in config.TRAINED_SYMBOLS


def test_known_micros_present():
    # the micros the bot advertises support for
    for micro in ("MNQ", "MES", "M2K", "MYM"):
        assert micro in config.MICRO_PARENT


def test_full_size_symbol_passes_through():
    for sym in config.TRAINED_SYMBOLS:
        assert config.base_symbol(sym) == sym


def test_unknown_symbol_passes_through():
    assert config.base_symbol("CL") == "CL"             # not a micro → unchanged
