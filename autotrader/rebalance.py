"""목표 비중 → 주문 목록.

신호가 바뀐 종목만 사고판다(백테스트와 동일). 보유 중인 종목은 비중이 조금 달라져도 그대로 둔다.
"""
from __future__ import annotations

import math

import pandas as pd

from autotrader.broker import Order


def make_orders(sig: pd.DataFrame, holdings: dict[str, int], budget: float, limit_buffer: float = 0.005,
                cash: float | None = None) -> list[Order]:
    """sig: strategy.signals() 결과, budget: 이 전략에 배정한 총액(USD), cash: 현재 현금(주면 매수 총액을 그 안으로 제한).

    - 보유 신호인데 안 가지고 있으면 → 목표 금액만큼 매수 (지정가 = 종가 × (1 + buffer))
    - 보유 신호가 아닌데 가지고 있으면 → 전량 매도 (지정가 = 종가 × (1 − buffer))
    """
    sells, buys = [], []
    for t, row in sig.iterrows():
        have = holdings.get(t, 0)
        if row["hold"] and have == 0:
            buys.append((t, round(row["close"] * (1 + limit_buffer), 2), budget * row["target_weight"]))
        elif not row["hold"] and have > 0:
            sells.append(Order(t, "sell", have, round(row["close"] * (1 - limit_buffer), 2)))
    if cash is not None and buys:
        avail = cash + sum(o.qty * o.price for o in sells) * 0.995  # 매도 대금(비용 감안) 포함
        want = sum(amt for *_, amt in buys)
        scale = min(1.0, avail / want) if want > 0 else 0.0
        buys = [(t, px, amt * scale) for t, px, amt in buys]
    orders = list(sells)  # 매도를 먼저 내서 현금을 확보
    for t, px, amt in buys:
        qty = math.floor(amt / px)
        if qty > 0:
            orders.append(Order(t, "buy", qty, px))
    return orders
