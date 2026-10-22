"""다구치 직교배열(L25) 생성과 분석 (주효과, ANOVA, 최적조건 예측)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def l25() -> np.ndarray:
    """L25(5^6) 직교배열. 행 (i, j) ∈ GF(5)², 열 = i, j, i+j, i+2j, i+3j, i+4j (mod 5).

    어떤 두 열을 골라도 25가지 수준 조합이 정확히 한 번씩 등장한다 (강도 2).
    """
    rows = []
    for i, j in itertools.product(range(5), repeat=2):
        rows.append([i, j] + [(i + k * j) % 5 for k in range(1, 5)])
    arr = np.array(rows)
    for a, b in itertools.combinations(range(arr.shape[1]), 2):  # 직교성 검증
        assert len({(x, y) for x, y in arr[:, [a, b]]}) == 25
    return arr


def design(factors: dict[str, list], array: np.ndarray) -> pd.DataFrame:
    """인자 dict(이름 → 5개 수준)를 배열 열에 차례로 배정. 남는 열은 오차 추정용 빈 열."""
    df = pd.DataFrame(index=range(len(array)))
    for col, (name, levels) in enumerate(factors.items()):
        df[name] = [levels[k] for k in array[:, col]]
    for col in range(len(factors), array.shape[1]):
        df[f"_e{col}"] = array[:, col]
    return df


def main_effects(df: pd.DataFrame, factors: list[str], y: str) -> dict[str, pd.Series]:
    return {f: df.groupby(f, sort=True)[y].mean() for f in factors}


def anova(df: pd.DataFrame, factors: list[str], y: str) -> pd.DataFrame:
    """주효과 ANOVA. 인자에 배정되지 않은 변동(빈 열 + 교호작용)을 오차로 풀링해 F검정."""
    grand = df[y].mean()
    total_ss = ((df[y] - grand) ** 2).sum()
    rows = []
    for f in factors:
        g = df.groupby(f)[y]
        ss = (g.count() * (g.mean() - grand) ** 2).sum()
        rows.append({"factor": f, "dof": g.ngroups - 1, "SS": ss})
    t = pd.DataFrame(rows).set_index("factor")
    err_ss = total_ss - t.SS.sum()
    err_dof = len(df) - 1 - t.dof.sum()
    mse = err_ss / err_dof if err_dof else np.nan
    t["MS"] = t.SS / t.dof
    t["F"] = t.MS / mse
    t["p_value"] = stats.f.sf(t.F, t.dof, err_dof)
    t["contrib_%"] = 100 * t.SS / total_ss
    t.loc["(error)"] = [err_dof, err_ss, mse, np.nan, np.nan, 100 * err_ss / total_ss]
    return t


def best_levels(effects: dict[str, pd.Series]) -> dict:
    return {f: s.idxmax() for f, s in effects.items()}


def predict(df: pd.DataFrame, effects: dict[str, pd.Series], levels: dict, y: str) -> float:
    """가법 모형 예측값: 전체평균 + Σ(최적수준 평균 − 전체평균)."""
    grand = df[y].mean()
    return grand + sum(effects[f][lv] - grand for f, lv in levels.items())
