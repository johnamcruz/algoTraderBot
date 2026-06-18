"""Broker contract/order resolution against a stubbed gateway. Front-month
selection drives rollover (wrong pick = trading a stale or wrong contract), and
working_stop_order is what the PPO exit reprices each bar."""
import broker


class StubClient(broker.TopstepXClient):
    def __init__(self, responses):
        super().__init__("u", "k")
        self._token = "t"
        self._responses = responses

    def _post(self, path, payload, auth=True):
        return self._responses[path]


def test_get_active_contract_picks_active_front_month():
    c = StubClient({"/Contract/search": {"contracts": [
        {"name": "NQM6", "activeContract": False, "tickSize": 0.25, "tickValue": 5},
        {"name": "NQU6", "activeContract": True, "tickSize": 0.25, "tickValue": 5},
    ]}})
    got = c.get_active_contract("NQ")
    assert got["name"] == "NQU6" and got["activeContract"] is True


def test_get_active_contract_ignores_other_symbols():
    # 'MNQU6'[:-2] == 'MNQ' != 'NQ' — a substring match must NOT pick the micro
    c = StubClient({"/Contract/search": {"contracts": [
        {"name": "MNQU6", "activeContract": True, "tickSize": 0.5, "tickValue": 0.5},
        {"name": "NQU6", "activeContract": True, "tickSize": 0.25, "tickValue": 5},
    ]}})
    assert c.get_active_contract("NQ")["name"] == "NQU6"


def test_get_contract_specs_returns_tick_size_and_value():
    c = StubClient({"/Contract/search": {"contracts": [
        {"name": "ESU6", "activeContract": True, "tickSize": 0.25, "tickValue": 12.5},
    ]}})
    assert c.get_contract_specs("ES") == (0.25, 12.5)


def test_get_active_contract_raises_when_none_active():
    c = StubClient({"/Contract/search": {"contracts": [
        {"name": "NQM6", "activeContract": False, "tickSize": 0.25, "tickValue": 5},
    ]}})
    try:
        c.get_active_contract("NQ")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_working_stop_order_filters_by_contract_type_and_status():
    c = StubClient({"/Order/searchOpen": {"orders": [
        {"id": 1, "contractId": "NQU6", "type": 1, "status": 1},   # limit (TP) — skip
        {"id": 2, "contractId": "ESU6", "type": 4, "status": 1},   # other contract — skip
        {"id": 3, "contractId": "NQU6", "type": 4, "status": 2},   # not working — skip
        {"id": 4, "contractId": "NQU6", "type": 4, "status": 1},   # the live stop ✓
    ]}})
    o = c.working_stop_order(7, "NQU6")
    assert o is not None and o["id"] == 4


def test_working_stop_order_none_when_no_match():
    c = StubClient({"/Order/searchOpen": {"orders": []}})
    assert c.working_stop_order(7, "NQU6") is None
