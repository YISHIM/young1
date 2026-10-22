"""최종 전략(EMA100 +1.5%/−4%, SPY>SMA225 필터)의 연도별 수익률 표와 그림.

실행:  python run_yearly_returns.py      (결과: results/yearly/)
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
OUT = ROOT / "results" / "yearly"
OUT.mkdir(parents=True, exist_ok=True)

etf = pd.read_csv(ROOT / "data/etf_adj_close.csv", index_col=0, parse_dates=True)
stk = pd.read_csv(ROOT / "data/stocks50_adj_close.csv", index_col=0, parse_dates=True).reindex(etf.index)
rf = pd.read_csv(ROOT / "data/risk_free.csv", index_col=0, parse_dates=True)["rf_annual_pct"]
rf_d = (1 + rf / 100) ** (1 / 252) - 1
idx = etf.index
spy_only = etf[["SPY"]]
B = {"ETF": Backtester(etf, rf_d, bench=etf["SPY"]), "SPY": Backtester(spy_only, rf_d, bench=etf["SPY"]),
     "STK": Backtester(stk, rf_d, bench=etf["SPY"])}


def levered(b: Backtester, pos: np.ndarray, m: float, spread: float) -> np.ndarray:
    """보유 비중 m배 (run_leverage_experiment.py 와 같은 계산). 차입 이자 = (무위험 + spread) × 차입분."""
    held = np.zeros_like(pos)
    held[2:] = pos[:-2]
    w = held * m
    trades = np.abs(np.diff(w, axis=0, prepend=0.0))
    sp = (1 + spread) ** (1 / 252) - 1
    r = w * np.nan_to_num(b.ret) + (1 - w) * b.rf[:, None] - np.maximum(w - 1, 0) * sp - trades * b.cost
    r[np.isnan(b.px)] = np.nan
    return r


def port(r: np.ndarray) -> pd.Series:
    out = np.full(len(r), np.nan)
    ok = ~np.all(np.isnan(r), axis=1)
    out[ok] = np.nanmean(r[ok], axis=1)
    return pd.Series(out, index=idx)


pos = {k: b.signal("EMA", 100, 1.5, 4.0, 225) for k, b in B.items()}
series = {
    "전략 ETF14": port(B["ETF"].returns(pos["ETF"])),
    "보유 ETF14": port(B["ETF"].buy_and_hold()),
    "전략 ETF14 ×1.6": port(levered(B["ETF"], pos["ETF"], 1.6, 0.015)),
    "전략 SPY만": port(B["SPY"].returns(pos["SPY"])),
    "보유 SPY": port(B["SPY"].buy_and_hold()),
    "전략 대형주50": port(B["STK"].returns(pos["STK"])),
    "보유 대형주50": port(B["STK"].buy_and_hold()),
}
df = pd.DataFrame(series).loc["2000-01-01":]
yearly = (1 + df.fillna(0)).groupby(df.index.year).prod() - 1
exposure = pd.Series(np.nanmean(pos["ETF"], axis=1), index=idx).loc["2000-01-01":]
yearly["ETF14 평균 투자비중"] = exposure.groupby(exposure.index.year).mean()
yearly.index.name = "연도"

last = idx[-1].date()
cagr = (1 + df.fillna(0)).prod() ** (252 / len(df)) - 1
summary = pd.DataFrame({
    "연평균(CAGR)": cagr,
    "플러스인 해": (yearly[df.columns] > 0).sum().astype(str) + f"/{len(yearly)}",
    "최악의 해": yearly[df.columns].min(),
    "최고의 해": yearly[df.columns].max(),
    "전략이 보유를 이긴 해": [
        f"{(yearly[c] > yearly['보유 ETF14']).sum()}/{len(yearly)}" if c.startswith("전략 ETF14") else
        f"{(yearly['전략 SPY만'] > yearly['보유 SPY']).sum()}/{len(yearly)}" if c == "전략 SPY만" else
        f"{(yearly['전략 대형주50'] > yearly['보유 대형주50']).sum()}/{len(yearly)}" if c == "전략 대형주50" else ""
        for c in df.columns],
})
summary.index.name = "전략"


def fmt(x):
    return "" if pd.isna(x) else f"{x * 100:+.1f}%"


lines = ["# 연도별 수익률", "",
         f"- 전략: EMA100 +1.5% 돌파 매수 / −4% 이탈 매도, SPY>SMA225 장기추세 필터. 다음날 종가 체결, 편도 0.1% 비용, 현금은 단기국채 이자",
         f"- ETF14·대형주50은 동일가중 포트폴리오. ×1.6 = 보유 시 1.6배 (차입금리 = 국채 + 1.5%)",
         f"- {idx[-1].year}년은 {last}까지 (연중)", "",
         "| 연도 | " + " | ".join(yearly.columns) + " |", "|" + "---|" * (len(yearly.columns) + 1)]
for y, row in yearly.iterrows():
    cells = [fmt(v) if c != "ETF14 평균 투자비중" else f"{v:.0%}" for c, v in row.items()]
    lines.append(f"| {y} | " + " | ".join(cells) + " |")
lines += ["", "## 요약", "", "| 전략 | 연평균(CAGR) | 플러스인 해 | 최악의 해 | 최고의 해 | 전략이 보유를 이긴 해 |", "|---|---|---|---|---|---|"]
for n, r in summary.iterrows():
    lines.append(f"| {n} | {fmt(r['연평균(CAGR)'])} | {r['플러스인 해']} | {fmt(r['최악의 해'])} | {fmt(r['최고의 해'])} | {r['전략이 보유를 이긴 해']} |")
lines += ["", "![연도별 수익률](yearly_returns.png)"]
(OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
yearly.to_csv(OUT / "yearly_returns.csv")

fig, ax = plt.subplots(figsize=(14, 4.5))
x = np.arange(len(yearly))
w = 0.38
ax.bar(x - w / 2, yearly["전략 ETF14"] * 100, w, label="Strategy (ETF14)", color="#2563eb")
ax.bar(x + w / 2, yearly["보유 ETF14"] * 100, w, label="Buy & Hold (ETF14)", color="#cbd5e1")
ax.set_xticks(x, yearly.index, rotation=60)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Annual return (%)")
ax.set_title(f"Annual returns 2000-{idx[-1].year} ({idx[-1].year} YTD)")
ax.grid(axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "yearly_returns.png", dpi=110)
print("\n".join(lines))
