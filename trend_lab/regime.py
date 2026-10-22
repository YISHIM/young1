"""장기추세 기울기 국면 정의와, 국면별로 다른 이평선을 고르는 '전환 규칙'의 고속 평가.

전환 규칙: t일 종가에 장기추세 기울기 국면 g[t]를 판정 → 그 국면에 배정된 이평선 L_g 의 신호를 따른다.
각 이평선의 신호(보유 상태)는 자기 과거만으로 계산되므로 미래 정보를 쓰지 않는다.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

TD = 252


def slope(bench: pd.Series, ma_len: int, lookback: int) -> pd.Series:
    """장기 이평선의 연율화 기울기: (MA_t / MA_{t-k} − 1) × 252/k."""
    ma = bench.rolling(ma_len, min_periods=ma_len).mean()
    return (ma / ma.shift(lookback) - 1) * TD / lookback


def thresholds(s: pd.Series, scheme: str, train: np.ndarray) -> list[float]:
    """국면 경계값. 반드시 학습 구간 데이터로만 정한다."""
    x = s[train].dropna()
    if scheme == "Q3":            # 학습구간 기울기 3분위 (하/중/상)
        return [float(x.quantile(1 / 3)), float(x.quantile(2 / 3))]
    if scheme == "Z2":            # 하락(<0) / 상승(≥0)
        return [0.0]
    if scheme == "Z3":            # 하락(<0) / 완만한 상승 / 가파른 상승 (양의 기울기 중앙값 기준)
        return [0.0, float(x[x > 0].median())]
    raise ValueError(scheme)


def classify(s: pd.Series, cuts: list[float]) -> np.ndarray:
    """국면 번호 0..K-1 (0 = 가장 낮은 기울기). 기울기 계산 전 초기 구간은 가운데(상승쪽) 국면으로 둔다."""
    g = np.searchsorted(np.asarray(cuts), s.to_numpy(), side="right")
    g[s.isna().to_numpy()] = len(cuts) // 2 if len(cuts) > 1 else 1
    return g


def combine(pos_by_len: list[np.ndarray], choice: tuple[int, ...], g: np.ndarray) -> np.ndarray:
    """국면별 선택 이평선의 보유 상태를 날짜별로 이어 붙인다."""
    out = np.empty_like(pos_by_len[0])
    for k, li in enumerate(choice):
        m = g == k
        out[m] = pos_by_len[li][m]
    return out


class FastSwitchEvaluator:
    """국면 전환 조합 수천~수만 개를 한 번에 평가 (셀별 합/제곱합 분해).

    전환 시점의 추가 매매비용만 무시한 근사이며, 최종 후보는 정확한 백테스트로 다시 확인한다.
    ex: [이평선 수, 일, 종목] 초과수익률(전략수익 − 무위험), cells: [일] 셀 구간 번호(−1=제외)
    """

    def __init__(self, ex: np.ndarray, cells: np.ndarray, n_cells: int):
        self.ex = ex
        valid = ~np.isnan(ex[0])
        self.valid = valid
        self.P = np.zeros((ex.shape[1], n_cells))
        ok = cells >= 0
        self.P[np.where(ok)[0], cells[ok]] = 1.0
        self.ex0 = np.where(valid[None], ex, 0.0)

    def sums(self, g_ret: np.ndarray, K: int):
        """g_ret: 각 '수익 발생일'에 적용된 국면 (신호일 국면을 2일 뒤로 민 것)."""
        n, s1, s2 = [], [], []
        for k in range(K):
            m = (g_ret == k)[:, None] & self.valid
            n.append(self.P.T @ m)
            e = np.transpose(self.ex0 * m[None], (0, 2, 1))  # [L,j,t]
            s1.append(np.transpose(e @ self.P, (0, 2, 1)))
            s2.append(np.transpose((e * e) @ self.P, (0, 2, 1)))
        return np.array(n), np.array(s1), np.array(s2)  # [K,c,j], [K,L,c,j], [K,L,c,j]

    @staticmethod
    def score(n, s1, s2, combos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """combos [M,K] → (로버스트 점수 = 셀 샤프 평균 − 표준편차, 셀 샤프 평균)."""
        K = combos.shape[1]
        N = n.sum(0)
        S1 = sum(s1[k][combos[:, k]] for k in range(K))
        S2 = sum(s2[k][combos[:, k]] for k in range(K))
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = S1 / N
            var = (S2 / N - mu * mu) * N / (N - 1)
            sh = (mu / np.sqrt(var) * np.sqrt(TD)).reshape(len(combos), -1)
        sh = np.where(np.isfinite(sh), sh, np.nan)
        mean = np.nanmean(sh, 1)
        return mean - np.nanstd(sh, 1, ddof=1), mean


def all_combos(n_len: int, K: int) -> np.ndarray:
    return np.array(list(itertools.product(range(n_len), repeat=K)))
