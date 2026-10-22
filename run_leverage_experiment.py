"""레버리지 검증: 장기추세 필터로 줄어든 낙폭만큼 투자 비중을 키우면 '그냥 보유'보다 나은가.

규칙: EMA100 +1.5% 돌파 매수 / −4% 이탈 매도, SPY>SMA225 필터 (다구치 최적). 보유 시 비중 m배(1.0~2.0).
차입분(m−1)은 단기국채 금리 + 가산금리(spread)로 이자를 낸다. 매매비용도 m배.

실행:  python run_leverage_experiment.py      (결과: results/leverage/)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trend_lab.backtest import Backtester

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).parent
OUT = ROOT / "results" / "leverage"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261024)

IS = ("2000-01-01", "2015-12-31")
OOS = ("2016-01-01", "2026-12-31")
FULL = ("2000-01-01", "2026-12-31")
LEV = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]
SPREADS = [0.005, 0.015, 0.03]   # 가산금리 연 0.5% / 1.5% / 3%
COST = 0.001
N_BOOT, BLOCK = 2000, 63

report: list[str] = []
IDX_CUR = None


def log(s: str = "") -> None:
    print(s, flush=True)
    report.append(s)


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    head = "| " + " | ".join([d.index.name or ""] + [str(c) for c in d.columns]) + " |"
    sep = "|" + "---|" * (len(d.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [str(v) for v in row]) + " |" for i, row in zip(d.index, d.values)]
    return "\n".join([head, sep, *body])


etf = pd.read_csv(ROOT / "data/etf_adj_close.csv", index_col=0, parse_dates=True)
stk = pd.read_csv(ROOT / "data/stocks50_adj_close.csv", index_col=0, parse_dates=True).reindex(etf.index)
rf = pd.read_csv(ROOT / "data/risk_free.csv", index_col=0, parse_dates=True)["rf_annual_pct"]
rf_d = (1 + rf / 100) ** (1 / 252) - 1
idx = etf.index
U = {"ETF 14종": Backtester(etf, rf_d, bench=etf["SPY"], cost=COST),
     "대형주 50종목": Backtester(stk, rf_d, bench=etf["SPY"], cost=COST)}


def levered(b: Backtester, pos: np.ndarray, m: float, spread: float) -> np.ndarray:
    """보유 비중 m배. 차입 이자 = (무위험 + spread) × 차입분."""
    held = np.zeros_like(pos)
    held[2:] = pos[:-2]
    w = held * m
    trades = np.abs(np.diff(w, axis=0, prepend=0.0))
    borrow = np.maximum(w - 1, 0)
    sp = (1 + spread) ** (1 / 252) - 1
    r = w * np.nan_to_num(b.ret) + (1 - w) * b.rf[:, None] - borrow * sp - trades * b.cost
    r[np.isnan(b.px)] = np.nan
    return r


def port(r):
    out = np.full(len(r), np.nan)
    ok = ~np.all(np.isnan(r), axis=1)
    out[ok] = np.nanmean(r[ok], axis=1)
    return out


def stats(pr: np.ndarray, rf: np.ndarray) -> dict:
    ok = ~np.isnan(pr)
    pr, rf = pr[ok], rf[ok]
    w = np.cumprod(1 + pr)
    ex = pr - rf
    yrs = len(pr) / 252
    s = pd.Series(pr, index=IDX_CUR[ok])
    yearly = (1 + s).groupby(s.index.year).prod() - 1
    return {"CAGR": w[-1] ** (1 / yrs) - 1, "변동성": pr.std(ddof=1) * np.sqrt(252),
            "샤프": ex.mean() / ex.std(ddof=1) * np.sqrt(252), "MDD": (w / np.maximum.accumulate(w) - 1).min(),
            "최악의 해": yearly.min(), "칼마(CAGR/|MDD|)": (w[-1] ** (1 / yrs) - 1) / abs((w / np.maximum.accumulate(w) - 1).min())}


def mask(span):
    return (idx >= span[0]) & (idx <= span[1])


log("# 레버리지 검증 리포트 — 낙폭이 줄어든 만큼 비중을 키우면?")
log()
log("- 기본 규칙: EMA100 +1.5% 돌파 매수 / −4% 이탈 매도, SPY>SMA225 장기추세 필터 (다구치 최적)")
log("- 보유 신호일 때 종목당 비중 m배 (예: 1.3 = 자기자본의 130%). 차입분 이자 = 단기국채 금리 + 가산금리. 매매비용 편도 0.1% × m")
log("- 동일가중 포트폴리오 기준. 학습 2000–2015 / 표본외 2016–2026")
log()

base_pos = {u: b.signal("EMA", 100, 1.5, 4.0, 225) for u, b in U.items()}
res = {}
for u, b in U.items():
    log(f"## {u}")
    log()
    for span_name, span in [("전체 2000–2026", FULL), ("학습 2000–2015", IS), ("표본외 2016–2026", OOS)]:
        m_ = mask(span)
        IDX_CUR = idx[m_]
        rows = {}
        for m in LEV:
            rows[f"전략 ×{m:.1f}"] = stats(port(levered(b, base_pos[u], m, 0.015))[m_], b.rf[m_])
        rows["그냥 보유 ×1.0"] = stats(port(b.buy_and_hold())[m_], b.rf[m_])
        t = pd.DataFrame(rows).T
        t.index.name = f"{span_name} (가산금리 1.5%)"
        res[(u, span_name)] = t
        log(md_table(t))
        log()

# 학습구간에서 레버리지 결정 → 표본외 확인 (사전 규칙: 학습구간 MDD가 그냥 보유 MDD의 절반을 넘지 않는 최대 m)
log("## 학습구간에서 레버리지를 정하고 표본외에서 확인")
log()
log("사전 규칙: **학습구간(2000–2015) 포트폴리오 MDD가 −30%를 넘지 않는 최대 배율** (그냥 보유는 −53~−57%). "
    "이 배율을 고정해서 표본외 2016–2026에 적용한다.")
log()
rows = []
for u, b in U.items():
    t_is = res[(u, "학습 2000–2015")]
    ok = [m for m in LEV if t_is.loc[f"전략 ×{m:.1f}", "MDD"] >= -0.30]
    m_sel = max(ok)
    for span_name in ["학습 2000–2015", "표본외 2016–2026", "전체 2000–2026"]:
        t = res[(u, span_name)]
        s, h = t.loc[f"전략 ×{m_sel:.1f}"], t.loc["그냥 보유 ×1.0"]
        rows.append({"투자대상·구간": f"{u} · {span_name}", "선택 배율": m_sel, "전략 CAGR": s["CAGR"], "보유 CAGR": h["CAGR"],
                     "전략 MDD": s["MDD"], "보유 MDD": h["MDD"], "전략 샤프": s["샤프"], "보유 샤프": h["샤프"],
                     "전략 최악의 해": s["최악의 해"], "보유 최악의 해": h["최악의 해"]})
sel = pd.DataFrame(rows).set_index("투자대상·구간")
log(md_table(sel))
log()

# 가산금리 민감도
log("## 가산금리(차입 비용) 민감도 — 전체 2000–2026, 배율 1.5")
log()
rows = []
for u, b in U.items():
    m_ = mask(FULL)
    IDX_CUR = idx[m_]
    for sp in SPREADS:
        s = stats(port(levered(b, base_pos[u], 1.5, sp))[m_], b.rf[m_])
        rows.append({"투자대상·가산금리": f"{u} · 연 {sp:.1%}", **s})
log(md_table(pd.DataFrame(rows).set_index("투자대상·가산금리")))
log()

# 통계 검정: 같은 변동성으로 맞춘 비교 (위험을 같게 했을 때 누가 더 버나)
log("## 통계 검정 — 위험(변동성)을 같게 맞추면 누가 더 버나")
log()
log("전략의 배율을 조정해 전체 구간 변동성을 그냥 보유와 같게 만든 뒤, 연 수익률 차이를 63일 블록 부트스트랩(2,000회)으로 검정.")
log()
rows = []
for u, b in U.items():
    m_ = mask(FULL)
    IDX_CUR = idx[m_]
    bh = port(b.buy_and_hold())[m_]
    vol_bh = np.nanstd(bh, ddof=1)
    lo, hi = 1.0, 4.0
    for _ in range(40):  # 이분법으로 변동성 맞추기
        mid = (lo + hi) / 2
        v = np.nanstd(port(levered(b, base_pos[u], mid, 0.015))[m_], ddof=1)
        lo, hi = (mid, hi) if v < vol_bh else (lo, mid)
    m_eq = (lo + hi) / 2
    st = port(levered(b, base_pos[u], m_eq, 0.015))[m_]
    ok = ~np.isnan(st) & ~np.isnan(bh)
    a, c = st[ok], bh[ok]
    L = len(a)
    nb = L // BLOCK + 1
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s0 = RNG.integers(0, L - BLOCK, nb)
        ix = (s0[:, None] + np.arange(BLOCK)).ravel()[:L]
        d[i] = (np.prod(1 + a[ix]) ** (252 / L) - 1) - (np.prod(1 + c[ix]) ** (252 / L) - 1)
    real = (np.prod(1 + a) ** (252 / L) - 1) - (np.prod(1 + c) ** (252 / L) - 1)
    sa, sc = stats(st, b.rf[m_]), stats(bh, b.rf[m_])
    rows.append({"투자대상": u, "변동성 맞춘 배율": m_eq, "전략 CAGR": sa["CAGR"], "보유 CAGR": sc["CAGR"],
                 "CAGR 차이": real, "90% 구간": f"{np.percentile(d, 5):+.2%} ~ {np.percentile(d, 95):+.2%}",
                 "p값(전략≤보유)": (d <= 0).mean(), "전략 MDD": sa["MDD"], "보유 MDD": sc["MDD"]})
log(md_table(pd.DataFrame(rows).set_index("투자대상")))
log()

# 그림: 배율별 CAGR vs MDD (전체 구간)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, u in zip(axes, U):
    t = res[(u, "전체 2000–2026")]
    s = t.drop("그냥 보유 ×1.0")
    ax.plot(-s["MDD"] * 100, s["CAGR"] * 100, "o-", label="Strategy x1.0 ... x2.0")
    for name, row in s.iterrows():
        ax.annotate(name.split("×")[1], (-row["MDD"] * 100, row["CAGR"] * 100), fontsize=8,
                    xytext=(4, -10), textcoords="offset points")
    h = t.loc["그냥 보유 ×1.0"]
    ax.plot(-h["MDD"] * 100, h["CAGR"] * 100, "r*", ms=14, label="Buy & Hold")
    ax.set_xlabel("Max drawdown (%)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title(f"{'ETF14' if 'ETF' in u else 'STOCK50'}: 2000-2026, spread 1.5%")
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(OUT / "leverage_frontier.png", dpi=110)
plt.close(fig)
log("![레버리지 곡선](leverage_frontier.png)")
(OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
print("saved →", OUT)
