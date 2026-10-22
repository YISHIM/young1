"""오늘의 목표 보유 상태 계산. 백테스트와 똑같은 함수(trend_lab.backtest.positions)를 써서 결과가 일치하도록 한다."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trend_lab.backtest import moving_average, positions, trend_filter


@dataclass
class Params:
    ma_type: str = "EMA"
    ma_len: int = 100
    buy_pct: float = 1.5
    sell_pct: float = 4.0
    trend_len: int = 225
    leverage: float = 1.0


def signals(price: pd.DataFrame, bench: pd.Series, p: Params) -> pd.DataFrame:
    """종목별 오늘(마지막 날짜) 상태표: 종가, 이평선, 매수선, 매도선, 보유 신호, 목표 비중."""
    ma = moving_average(price, p.ma_len, p.ma_type)
    tok = trend_filter(bench.reindex(price.index), p.trend_len)
    pos = positions(price.to_numpy(), ma.to_numpy(), p.buy_pct / 100, p.sell_pct / 100, tok)
    last = price.index[-1]
    n = price.shape[1]
    hold = pos[-1].astype(bool)
    out = pd.DataFrame({
        "close": price.iloc[-1],
        "ma": ma.iloc[-1],
        "buy_line": ma.iloc[-1] * (1 + p.buy_pct / 100),
        "sell_line": ma.iloc[-1] * (1 - p.sell_pct / 100),
        "hold": hold,
        "changed": pos[-1] != (pos[-2] if len(pos) > 1 else np.zeros(n)),
    })
    out["target_weight"] = np.where(hold, p.leverage / n, 0.0)
    out.attrs["date"] = last
    out.attrs["trend_ok"] = bool(tok[-1])
    bench_ma = bench.rolling(p.trend_len).mean()
    out.attrs["bench_close"] = float(bench.iloc[-1])
    out.attrs["bench_trend_line"] = float(bench_ma.iloc[-1]) if p.trend_len else float("nan")
    return out
