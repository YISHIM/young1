"""이동평균 돌파/이탈 + 장기추세 필터 백테스트 엔진 (벡터화).

규칙
----
- 매수: 종가 > MA × (1 + buy_pct)   그리고  장기추세 필터 ON
- 매도: 종가 < MA × (1 − sell_pct)  또는    장기추세 필터 OFF
- 그 사이(밴드 안): 직전 상태 유지 (히스테리시스 → 잦은 휩쏘 매매 억제)
- 장기추세 필터: 기준종목(SPY) 종가 > 기준종목 SMA(trend_len).  trend_len=0 이면 필터 없음
- 체결: 신호가 난 다음 거래일 종가에 체결 (보수적, 미래 정보 사용 없음)
- 현금 보유 시 무위험이자(단기국채) 수취, 매매 1회(편도)당 비용 cost 차감
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def moving_average(price: pd.DataFrame, length: int, kind: str) -> pd.DataFrame:
    if kind == "SMA":
        return price.rolling(length, min_periods=length).mean()
    if kind == "EMA":
        ema = price.ewm(span=length, adjust=False, min_periods=length).mean()
        return ema
    raise ValueError(kind)


class MACache:
    """같은 이평선을 반복 계산하지 않도록 캐시."""

    def __init__(self, price: pd.DataFrame):
        self.price = price
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def get(self, length: int, kind: str) -> np.ndarray:
        key = (kind, length)
        if key not in self._cache:
            self._cache[key] = moving_average(self.price, length, kind).to_numpy()
        return self._cache[key]


def trend_filter(bench: pd.Series, trend_len: int) -> np.ndarray:
    """기준종목이 장기 이평선 위에 있으면 True (장기 상승추세)."""
    if trend_len == 0:
        return np.ones(len(bench), dtype=bool)
    ma = bench.rolling(trend_len, min_periods=trend_len).mean()
    return (bench > ma).to_numpy()


def positions(px: np.ndarray, ma: np.ndarray, buy_pct: float, sell_pct: float,
              trend_ok: np.ndarray) -> np.ndarray:
    """종목별 보유 상태(1=보유, 0=현금) 행렬 [일 × 종목]."""
    valid = ~np.isnan(ma) & ~np.isnan(px)
    tok = trend_ok[:, None]
    entry = valid & tok & (px > ma * (1 + buy_pct))
    exit_ = (valid & (px < ma * (1 - sell_pct))) | ~tok | ~valid
    state = np.where(exit_, 0.0, np.where(entry, 1.0, np.nan))  # 매도 우선
    state = pd.DataFrame(state).ffill().fillna(0.0).to_numpy()
    return state


def strategy_returns(px: np.ndarray, ret: np.ndarray, rf: np.ndarray, pos: np.ndarray,
                     cost: float) -> np.ndarray:
    """일별 전략 수익률. pos[t]는 t일 종가 신호 → t+1일 종가 체결 → t+2일부터 수익."""
    held = np.zeros_like(pos)
    held[2:] = pos[:-2]
    trades = np.abs(np.diff(held, axis=0, prepend=0.0))
    r = held * np.nan_to_num(ret) + (1 - held) * rf[:, None] - trades * cost
    r[np.isnan(px)] = np.nan  # 상장 전 구간 제외
    return r


def metrics(r: np.ndarray, rf: np.ndarray) -> dict[str, np.ndarray]:
    """열(종목)별 CAGR, 연변동성, 샤프, MDD. NaN 구간 무시."""
    mask = ~np.isnan(r)
    n = mask.sum(axis=0)
    rr = np.where(mask, r, 0.0)
    wealth = np.cumprod(1 + rr, axis=0)
    years = n / TRADING_DAYS
    cagr = wealth[-1] ** (1 / np.maximum(years, 1e-9)) - 1
    ex = np.where(mask, r - rf[:, None], np.nan)
    mu = np.nanmean(ex, axis=0)
    sd = np.nanstd(ex, axis=0, ddof=1)
    sharpe = mu / sd * np.sqrt(TRADING_DAYS)
    peak = np.maximum.accumulate(wealth, axis=0)
    mdd = (wealth / peak - 1).min(axis=0)
    return {"cagr": cagr, "vol": sd * np.sqrt(TRADING_DAYS), "sharpe": sharpe, "mdd": mdd,
            "years": years}


class Backtester:
    def __init__(self, price: pd.DataFrame, rf_daily: pd.Series, bench: str = "SPY",
                 cost: float = 0.001):
        self.price = price
        self.px = price.to_numpy()
        self.ret = price.pct_change(fill_method=None).to_numpy()
        self.rf = rf_daily.reindex(price.index).ffill().fillna(0.0).to_numpy()
        self.bench = price[bench]
        self.cost = cost
        self.ma = MACache(price)
        self._trend: dict[int, np.ndarray] = {}

    def trend(self, trend_len: int) -> np.ndarray:
        if trend_len not in self._trend:
            self._trend[trend_len] = trend_filter(self.bench, trend_len)
        return self._trend[trend_len]

    def run(self, ma_type: str, ma_len: int, buy_pct: float, sell_pct: float,
            trend_len: int) -> np.ndarray:
        pos = positions(self.px, self.ma.get(ma_len, ma_type), buy_pct / 100, sell_pct / 100,
                        self.trend(trend_len))
        return strategy_returns(self.px, self.ret, self.rf, pos, self.cost)

    def buy_and_hold(self) -> np.ndarray:
        return self.ret.copy()

    def evaluate(self, r: np.ndarray, periods: list[tuple[str, str]]) -> dict[str, np.ndarray]:
        """기간 × 종목 셀별 성과 → {지표: [기간, 종목] 배열}."""
        idx = self.price.index
        out: dict[str, list] = {}
        for start, end in periods:
            sl = (idx >= start) & (idx <= end)
            m = metrics(r[sl], self.rf[sl])
            for k, v in m.items():
                out.setdefault(k, []).append(v)
        return {k: np.array(v) for k, v in out.items()}
