"""자동매매 프로그램이 백테스트와 같은 매매를 하는지 검증."""
from __future__ import annotations

import numpy as np
import pandas as pd

from autotrader.broker import Order, PaperBroker
from autotrader.data import load_prices
from autotrader.rebalance import make_orders
from autotrader.strategy import Params, signals
from trend_lab.backtest import Backtester

TICKERS = ["SPY", "QQQ", "XLE", "XLU"]


def test_replay_matches_backtest(tmp_path):
    """2019–2022을 하루씩 재생: 매일 종가로 신호 → 다음날 종가에 가상 체결. 백테스트 보유 상태와 일치해야 한다."""
    price, bench = load_prices(TICKERS, "SPY", source="csv")
    p = Params()
    bt = Backtester(price, pd.Series(0.0, index=price.index), bench=bench, cost=0.001)
    bt_pos = pd.DataFrame(bt.signal(p.ma_type, p.ma_len, p.buy_pct, p.sell_pct, p.trend_len),
                          index=price.index, columns=TICKERS)
    broker = PaperBroker(tmp_path / "paper.json", initial_cash=1_000_000, cost=0.001)
    days = price.loc["2019-01-01":"2022-12-31"].index
    pending: list[Order] = []
    for d in days:
        i = price.index.get_loc(d)
        for o in pending:  # 어제 신호 → 오늘 종가 체결
            res = broker.place(Order(o.ticker, o.side, o.qty, float(price.iloc[i][o.ticker])))
            assert res["ok"], res
        sig = signals(price.iloc[: i + 1], bench.iloc[: i + 1], p)
        assert (sig["hold"].to_numpy() == bt_pos.iloc[i].to_numpy().astype(bool)).all(), d
        pending = make_orders(sig, broker.positions(), 1_000_000, limit_buffer=0.03, cash=broker.cash())
        held_now = {t for t, q in broker.positions().items() if q > 0}
        expected = {t for t in TICKERS if bt_pos.iloc[i - 1][t] == 1} if i > 0 else set()
        assert held_now == expected, (d, held_now, expected)


def test_orders_sell_first_and_qty():
    sig = pd.DataFrame({"close": [100.0, 50.0], "hold": [True, False], "target_weight": [0.5, 0.0]},
                       index=["AAA", "BBB"])
    orders = make_orders(sig, {"BBB": 7}, budget=10_000, limit_buffer=0.0)
    assert [o.side for o in orders] == ["sell", "buy"]
    assert orders[0].qty == 7 and orders[1].qty == 50
    # 현금이 부족하면 매수 수량을 줄인다 (현금 1,000 + 매도대금 350×0.995)
    small = make_orders(sig, {"BBB": 7}, budget=10_000, limit_buffer=0.0, cash=1000)
    assert small[1].qty == 13


def test_paper_broker_rejects_overspend(tmp_path):
    b = PaperBroker(tmp_path / "s.json", initial_cash=1000)
    assert not b.place(Order("SPY", "buy", 10, 200.0))["ok"]
    assert b.place(Order("SPY", "buy", 4, 200.0))["ok"]
    assert b.positions() == {"SPY": 4}
    assert np.isclose(b.cash(), 1000 - 800 - 0.8)
