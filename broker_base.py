#!/usr/bin/env python3
"""broker_base.py — the broker interface.

Defines the contract every broker must satisfy so the bot stays broker-agnostic.
To add a broker (e.g. Rithmic), implement `BrokerClient` and register it in
`broker.make_broker()` — nothing in the strategy/exit/bar-loop changes.

  OrderRouter   order + position routing. Implemented by live brokers AND the
                backtest SimBroker (all the bar loop, bot.handle_bar, needs).
  BrokerClient  a full live broker: account + market data + contract specs on
                top of order routing.

Canonical encodings every implementation must honor (so the bot never depends on
a specific broker's codes):
  SIDE           order side passed to place_*: 0 = buy, 1 = sell
  POSITION_LONG  the value of open_position()['type'] for a long position
"""
from abc import ABC, abstractmethod
from typing import Optional

SIDE = {"BUY": 0, "SELL": 1}
POSITION_LONG = 1


class OrderRouter(ABC):
    """Order + position routing — the minimal surface the bar loop drives."""

    @abstractmethod
    def open_position(self, account_id, contract_id) -> Optional[dict]:
        """The open position for the contract, or None. The dict carries at
        least `size`, `averagePrice`, and `type` (== POSITION_LONG for a long)."""

    @abstractmethod
    def place_market_with_brackets(self, account_id, contract_id, *, side, size,
                                   stop_ticks, target_ticks) -> dict:
        """Market entry with an OCO stop + take-profit (distances in ticks)."""

    @abstractmethod
    def place_market_with_stop(self, account_id, contract_id, *, side, size,
                               stop_ticks) -> dict:
        """Market entry with a protective stop only (the PPO trail manages it)."""

    @abstractmethod
    def place_market_with_trail(self, account_id, contract_id, *, side, size,
                                trail_ticks) -> dict:
        """Market entry with a broker-native trailing stop (follow distance in ticks)."""

    @abstractmethod
    def working_stop_order(self, account_id, contract_id) -> Optional[dict]:
        """The working protective stop order (dict with `id` and `stopPrice`), or None."""

    @abstractmethod
    def modify_stop_price(self, account_id, order_id, stop_price) -> dict:
        """Reprice a working stop order to `stop_price`."""

    @abstractmethod
    def modify_trail_price(self, account_id, order_id, trail_price) -> dict:
        """Tighten a native trailing stop's follow distance (a price distance)."""

    @abstractmethod
    def cancel_order(self, account_id, order_id) -> dict:
        """Cancel a working order."""

    @abstractmethod
    def close_position(self, account_id, contract_id, price=None) -> dict:
        """Flatten the position at market AND cancel any resting bracket orders
        for the contract (a market close doesn't fire the OCO, so the protective
        stop/TP would otherwise orphan and could fill into a naked position).
        `price` is an optional fill hint used by the backtest sim; live brokers
        close at market and ignore it."""


class BrokerClient(OrderRouter):
    """A full live broker: connection, account, market data and contract specs
    on top of order routing. Implement this to add a broker."""

    @abstractmethod
    def authenticate(self) -> None:
        """Establish an authenticated session."""

    @abstractmethod
    def pick_account(self, selector: str = "") -> dict:
        """Choose a tradable account (by id/name, or the first when blank)."""

    @abstractmethod
    def get_active_contract(self, symbol, live: bool = False) -> dict:
        """The active contract for `symbol` (carries tickSize / tickValue)."""

    @abstractmethod
    def get_contract_specs(self, symbol, live: bool = False):
        """(tick_size, tick_value) for `symbol`'s active contract."""

    @abstractmethod
    def get_bars(self, contract_id, minutes, limit: int = 300):
        """Recent OHLCV bars as a DataFrame (columns: time, open, high, low,
        close, volume)."""
