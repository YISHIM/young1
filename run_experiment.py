"""이동평균 돌파 전략 × 장기추세 필터 — 다구치 실험계획 최적화 + 전수조사 검증 + 표본외 검증.

실행:  python run_experiment.py        (결과: results/)
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from trend_lab import taguchi
from trend_lab.backtest import Backtester

ROOT = Path(__file__).parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

UNIVERSE = ["SPY", "QQQ", "DIA", "IWM", "MDY",
            "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
BROAD = ["SPY", "QQQ", "DIA", "IWM", "MDY"]
# 학습(최적화) 구간을 3개 국면으로 나눠 '잡음 인자'로 사용 → 국면이 바뀌어도 통하는 조건을 찾는다
IS_PERIODS = [("2000-01-01", "2005-12-31"),   # 닷컴 붕괴 + 회복
              ("2006-01-01", "2010-12-31"),   # 금융위기
              ("2011-01-01", "2015-12-31")]   # 완만한 상승장
OOS = ("2016-01-01", "2026-12-31")            # 표본외 검증: 최적화에 한 번도 쓰지 않은 구간
COST = 0.001                                  # 편도 0.1% (수수료+슬리피지)

report: list[str] = []


def log(s: str = "") -> None:
    print(s)
    report.append(s)


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    head = "| " + " | ".join([d.index.name or ""] + [str(c) for c in d.columns]) + " |"
    sep = "|" + "---|" * (len(d.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [str(v) for v in row]) + " |"
            for i, row in zip(d.index, d.values)]
    return "\n".join([head, sep, *body])


# ---------------------------------------------------------------- 데이터
price = pd.read_csv(ROOT / "data/etf_adj_close.csv", index_col=0, parse_dates=True)[UNIVERSE]
rf = pd.read_csv(ROOT / "data/risk_free.csv", index_col=0, parse_dates=True)["rf_annual_pct"]
rf_daily = (1 + rf / 100) ** (1 / 252) - 1
bt = Backtester(price, rf_daily, bench="SPY", cost=COST)

log("# 이동평균 돌파 전략 최적화 리포트 (다구치 실험계획법)")
log()
log(f"- 데이터: 미국 ETF 14종 수정주가(배당·분할 반영) {price.index[0].date()} ~ {price.index[-1].date()}")
log(f"- 학습 구간(잡음 인자 3국면): {', '.join(f'{a[:4]}–{b[:4]}' for a, b in IS_PERIODS)}")
log(f"- 표본외 검증 구간: {OOS[0][:4]} ~ {price.index[-1].date()} (최적화에 미사용)")
log(f"- 비용: 편도 {COST:.1%}, 현금 보유 시 단기국채 이자 수취, 신호 다음날 종가 체결")
log()

# ---------------------------------------------------------------- 1. 기준종목 선정
log("## 1. 장기추세 기준종목 선정 — '시장 전체를 가장 잘 대변하는 종목'")
log()
is_ret = price.pct_change(fill_method=None).loc["2000-01-01":"2015-12-31"]
ew = is_ret.mean(axis=1)  # 투자대상군 동일가중 수익률 = '시장 전체'
rows = []
for t in BROAD:
    x = pd.concat([is_ret[t], ew], axis=1).dropna()
    r = x.corr().iloc[0, 1]
    rows.append({"ticker": t, "상관계수": r, "설명력 R²": r * r})
sel = pd.DataFrame(rows).set_index("ticker").sort_values("설명력 R²", ascending=False)
BENCH = sel.index[0]
log("투자대상군 14종 동일가중 일간수익률과 각 지수 ETF의 상관관계(2000–2015):")
log()
log(md_table(sel))
log()
log(f"→ 기준종목: **{BENCH}** (투자대상군 전체 움직임을 가장 잘 설명)")
log()
bt = Backtester(price, rf_daily, bench=BENCH, cost=COST)

# ---------------------------------------------------------------- 공통 평가 함수
N_CELLS = len(IS_PERIODS) * len(UNIVERSE)


def evaluate(cfg: dict) -> dict:
    r = bt.run(cfg["ma_type"], int(cfg["ma_len"]), cfg["buy_pct"], cfg["sell_pct"],
               int(cfg["trend_len"]))
    m = bt.evaluate(r, IS_PERIODS + [OOS])
    sh_is = m["sharpe"][:3].ravel()
    sh_is = sh_is[~np.isnan(sh_is)]
    return {
        "sharpe": sh_is.mean(),                       # 평균 샤프 (42개 셀)
        "sn": sh_is.mean() - sh_is.std(ddof=1),       # 로버스트 점수 = 평균 − 표준편차
        "cagr": np.nanmean(m["cagr"][:3]),
        "mdd": np.nanmean(m["mdd"][:3]),
        "oos_sharpe": np.nanmean(m["sharpe"][3]),
        "oos_cagr": np.nanmean(m["cagr"][3]),
        "oos_mdd": np.nanmean(m["mdd"][3]),
    }


def run_design(df: pd.DataFrame) -> pd.DataFrame:
    res = [evaluate(row) for row in df.to_dict("records")]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(res)], axis=1)


FACTORS = ["ma_type", "ma_len", "buy_pct", "sell_pct", "trend_len"]
LABEL = {"ma_type": "이평 종류", "ma_len": "이평 기간(일)", "buy_pct": "매수 돌파 %",
         "sell_pct": "매도 이탈 %", "trend_len": "장기추세 SMA(0=없음)"}
LABEL_EN = {"ma_type": "MA type", "ma_len": "MA length (days)", "buy_pct": "Buy breakout %",
            "sell_pct": "Sell breakdown %", "trend_len": "Trend SMA on SPY (0=off)"}
Y = "sn"

# ---------------------------------------------------------------- 2. 1단계: 넓게 탐색 (L25 × 이평종류 2 = 50회)
log("## 2. 1단계 실험 — L25 직교배열 × 이평 종류(SMA/EMA) = 50회")
log()
lv1 = {"ma_len": [10, 30, 50, 75, 100],
       "buy_pct": [0, 1, 2, 3, 5],
       "sell_pct": [0, 1, 2, 3, 5],
       "trend_len": [0, 100, 150, 200, 250]}
L = taguchi.l25()
base = taguchi.design(lv1, L)
d1 = pd.concat([base.assign(ma_type=t) for t in ["SMA", "EMA"]], ignore_index=True)
t0 = time.time()
r1 = run_design(d1)
log(f"인자와 수준: " + "; ".join(f"{LABEL[k]} {v}" for k, v in lv1.items()) + "; 이평 종류 [SMA, EMA]")
log()
log(f"반응값: 학습구간 3국면 × 14종목 = {N_CELLS}개 셀의 샤프지수 → "
    "**로버스트 점수 = 평균 − 표준편차** (다구치 SN비의 취지: 잡음(국면·종목)이 바뀌어도 성과가 흔들리지 않는 조건)")
log(f"(50회 소요 {time.time() - t0:.1f}초)")
log()
eff1 = taguchi.main_effects(r1, FACTORS, Y)
an1 = taguchi.anova(r1, FACTORS, Y)
an1.index = [LABEL.get(i, i) for i in an1.index]
log("### 분산분석(ANOVA) — 어떤 인자가 성과를 좌우하나")
log()
log(md_table(an1[["dof", "SS", "F", "p_value", "contrib_%"]], "{:.4f}"))
log()
log("### 수준별 평균 로버스트 점수 (주효과)")
log()
for f in FACTORS:
    s = eff1[f]
    log(f"- {LABEL[f]}: " + ", ".join(f"{k}→{v:.3f}" for k, v in s.items()) + f"  ⇒ 최적 **{s.idxmax()}**")
log()
best1 = taguchi.best_levels(eff1)
pred1 = taguchi.predict(r1, eff1, best1, Y)

# 주효과 그래프
fig, axes = plt.subplots(1, 5, figsize=(17, 3.4), sharey=True)
for ax, f in zip(axes, FACTORS):
    s = eff1[f]
    ax.plot([str(i) for i in s.index], s.values, "o-", color="#2563eb")
    ax.axhline(r1[Y].mean(), color="gray", ls="--", lw=0.8)
    ax.set_title(LABEL_EN[f])
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Robust score (mean-std Sharpe)")
fig.suptitle("Stage 1 main effects (Taguchi L25 x MA type, 50 runs)")
fig.tight_layout()
fig.savefig(OUT / "stage1_main_effects.png", dpi=110)
plt.close(fig)


# ---------------------------------------------------------------- 3. 2단계: 최적점 주변 정밀 탐색 (L25 = 25회)
def around(center, step, lo, hi, n=5):
    vals = [center + step * k for k in range(-(n // 2), n // 2 + 1)]
    shift = max(0, lo - vals[0]) or min(0, hi - vals[-1])
    return [float(round(v + shift, 2)) if isinstance(step, float) else int(v + shift) for v in vals]


lv2 = {"ma_len": around(best1["ma_len"], 5, 10, 100),
       "buy_pct": around(best1["buy_pct"], 0.5, 0, 6),
       "sell_pct": around(best1["sell_pct"], 0.5, 0, 6),
       "trend_len": around(best1["trend_len"], 25, 0, 300) if best1["trend_len"] else [0, 50, 100, 150, 200]}
d2 = taguchi.design(lv2, L).assign(ma_type=best1["ma_type"])
r2 = run_design(d2)
F2 = ["ma_len", "buy_pct", "sell_pct", "trend_len"]
eff2 = taguchi.main_effects(r2, F2, Y)
an2 = taguchi.anova(r2, F2, Y)
an2.index = [LABEL.get(i, i) for i in an2.index]
best2 = {"ma_type": best1["ma_type"], **taguchi.best_levels(eff2)}
pred2 = taguchi.predict(r2, eff2, {k: best2[k] for k in F2}, Y)
log(f"## 3. 2단계 실험 — 1단계 최적점 주변 정밀 탐색 (L25 = 25회, 이평 종류 {best1['ma_type']} 고정)")
log()
log("인자와 수준: " + "; ".join(f"{LABEL[k]} {v}" for k, v in lv2.items()))
log()
log(md_table(an2[["dof", "SS", "F", "p_value", "contrib_%"]], "{:.4f}"))
log()
for f in F2:
    s = eff2[f]
    log(f"- {LABEL[f]}: " + ", ".join(f"{k}→{v:.3f}" for k, v in s.items()) + f"  ⇒ 최적 **{s.idxmax()}**")
log()

# ---------------------------------------------------------------- 4. 확인 실험
conf = evaluate(best2)
log("## 4. 확인 실험 (다구치 최적조건을 실제로 돌려 예측과 비교)")
log()
log(f"- 최적조건: " + ", ".join(f"{LABEL[k]}={best2[k]}" for k in FACTORS))
log(f"- 가법모형 예측 로버스트 점수 {pred2:.3f} vs 실측 {conf['sn']:.3f}  "
    f"(차이가 작으면 인자 간 교호작용이 작아 다구치 가정이 성립)")
log(f"- 학습구간 평균 샤프 {conf['sharpe']:.3f}, 평균 CAGR {conf['cagr']:.2%}, 평균 MDD {conf['mdd']:.2%}")
bh = bt.evaluate(bt.buy_and_hold(), IS_PERIODS)
log(f"- (비교) 같은 구간 매수 후 보유: 평균 샤프 {np.nanmean(bh['sharpe']):.3f}, "
    f"평균 CAGR {np.nanmean(bh['cagr']):.2%}, 평균 MDD {np.nanmean(bh['mdd']):.2%}")
log()

# ---------------------------------------------------------------- 5. 전수조사로 다구치 결과 검증
log("## 5. 검증 — 전수조사(Full grid)와 비교: 75회 실험으로 얼마나 최적에 가까웠나")
log()
grid_lv = {"ma_type": ["SMA", "EMA"],
           "ma_len": list(range(10, 101, 5)),
           "buy_pct": [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5],
           "sell_pct": [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5],
           "trend_len": [0, 100, 125, 150, 175, 200, 225, 250]}
grid_cache = OUT / f"full_grid_{price.index[-1].date()}.csv"  # 데이터 갱신 시 자동 재계산
if grid_cache.exists():
    G = pd.read_csv(grid_cache)
else:
    combos = list(itertools.product(*grid_lv.values()))
    t0 = time.time()
    rows = []
    for i, c in enumerate(combos):
        cfg = dict(zip(grid_lv, c))
        rows.append({**cfg, **evaluate(cfg)})
        if i % 2000 == 0:
            print(f"  grid {i}/{len(combos)}  {time.time() - t0:.0f}s", flush=True)
    G = pd.DataFrame(rows)
    G.to_csv(grid_cache, index=False)
G = G.sort_values(Y, ascending=False).reset_index(drop=True)
n_grid = len(G)
gbest = G.iloc[0]
rank = int((G[Y] > conf["sn"] + 1e-12).sum()) + 1
log(f"- 전수조사 조합 수: **{n_grid:,}회** vs 다구치 **75회** (약 {n_grid / 75:.0f}배 절약)")
log(f"- 다구치 최적조건 순위: {n_grid:,}개 중 **{rank}위** (상위 {rank / n_grid:.2%})")
log(f"- 전수조사 1위: " + ", ".join(f"{LABEL[k]}={gbest[k]}" for k in FACTORS) +
    f" → 로버스트 점수 {gbest[Y]:.3f} (다구치 {conf['sn']:.3f})")
log()
log("전수조사 상위 10개 조합 (학습구간 기준):")
log()
top = G.head(10)[FACTORS + ["sn", "sharpe", "cagr", "mdd"]].copy()
top.index = range(1, 11)
top.index.name = "순위"
log(md_table(top))
log()
# 과최적화 점검: 학습구간 성적과 표본외 성적의 순위상관
rho, p = stats.spearmanr(G["sn"], G["oos_sharpe"])
log(f"- 과최적화 점검: 학습구간 로버스트 점수 vs 표본외 샤프 순위상관 ρ = {rho:.3f} (p={p:.1e}). "
    "양(+)이면 학습구간에서 좋던 조건이 미래에도 대체로 좋았다는 뜻")
q = pd.qcut(G["sn"], 5, labels=["하위20%", "20-40%", "40-60%", "60-80%", "상위20%"])
qt = G.groupby(q, observed=True)[["sn", "oos_sharpe", "oos_cagr", "oos_mdd"]].mean()
qt.index.name = "학습구간 성적 5분위"
log()
log(md_table(qt))
log()

# 히트맵: 이평 기간 × 장기추세 (나머지 인자는 다구치 최적값 고정)
sub = G[(G.ma_type == best2["ma_type"]) & np.isclose(G.buy_pct, best2["buy_pct"]) &
        np.isclose(G.sell_pct, best2["sell_pct"])]
if len(sub):
    hm = sub.pivot_table(index="trend_len", columns="ma_len", values=Y)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    im = ax.imshow(hm.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(hm.columns)), hm.columns)
    ax.set_yticks(range(len(hm.index)), hm.index)
    ax.set_xlabel("MA length (days)")
    ax.set_ylabel("Trend filter SMA on " + BENCH)
    ax.set_title(f"Robust score grid ({best2['ma_type']}, buy {best2['buy_pct']}%, sell {best2['sell_pct']}%)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT / "grid_heatmap.png", dpi=110)
    plt.close(fig)

# ---------------------------------------------------------------- 6. 표본외 검증
log(f"## 6. 표본외 검증 {OOS[0][:4]}–{price.index[-1].year} (최적화에 쓰지 않은 미래 구간)")
log()
cands = {
    "다구치 최적": best2,
    "전수조사 1위": {k: gbest[k] for k in FACTORS},
    "기본형 SMA200 단독(필터 없음)": {"ma_type": "SMA", "ma_len": 200, "buy_pct": 0, "sell_pct": 0, "trend_len": 0},
}
idx = price.index
oos_sl = (idx >= OOS[0]) & (idx <= OOS[1])
rets = {name: bt.run(c["ma_type"], int(c["ma_len"]), c["buy_pct"], c["sell_pct"], int(c["trend_len"]))
        for name, c in cands.items()}
rets["매수 후 보유(B&H)"] = bt.buy_and_hold()
per = {}
summ = []
for name, r in rets.items():
    m = bt.evaluate(r, [OOS])
    per[name] = pd.DataFrame({k: m[k][0] for k in ["cagr", "sharpe", "mdd"]}, index=UNIVERSE)
    pr = np.nanmean(r[oos_sl], axis=1)  # 14종 동일가중 포트폴리오
    w = np.cumprod(1 + pr)
    pf = {"cagr": w[-1] ** (252 / len(pr)) - 1,
          "sharpe": (pr - bt.rf[oos_sl]).mean() / (pr - bt.rf[oos_sl]).std(ddof=1) * np.sqrt(252),
          "mdd": (w / np.maximum.accumulate(w) - 1).min()}
    summ.append({"전략": name, "종목평균 CAGR": m["cagr"][0].mean(), "종목평균 샤프": m["sharpe"][0].mean(),
                 "종목평균 MDD": m["mdd"][0].mean(), "포트폴리오 CAGR": pf["cagr"],
                 "포트폴리오 샤프": pf["sharpe"], "포트폴리오 MDD": pf["mdd"]})
summ = pd.DataFrame(summ).set_index("전략")
log("14종 각각에 전략을 적용한 평균, 그리고 14종 동일가중 포트폴리오 성과:")
log()
log(md_table(summ))
log()
cmp = pd.DataFrame({f"{n} {k}": per[n][k] for n in ["다구치 최적", "매수 후 보유(B&H)"] for k in ["cagr", "sharpe", "mdd"]})
cmp.index.name = "종목"
log("종목별 (다구치 최적 vs 매수 후 보유):")
log()
log(md_table(cmp))
log()

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
names_en = {"다구치 최적": "Taguchi optimum", "전수조사 1위": "Full-grid best",
            "기본형 SMA200 단독(필터 없음)": "SMA200 baseline", "매수 후 보유(B&H)": "Buy & Hold"}
for name, r in rets.items():
    spy = np.nan_to_num(r[oos_sl, UNIVERSE.index("SPY")])
    axes[0].plot(idx[oos_sl], np.cumprod(1 + spy), label=names_en[name])
    pr = np.nanmean(r[oos_sl], axis=1)
    axes[1].plot(idx[oos_sl], np.cumprod(1 + pr), label=names_en[name])
axes[0].set_title("SPY only (out-of-sample)")
axes[1].set_title("14-ETF equal-weight portfolio (out-of-sample)")
for ax in axes:
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(OUT / "oos_equity.png", dpi=110)
plt.close(fig)

log("![1단계 주효과](stage1_main_effects.png)")
log()
log("![전수조사 히트맵](grid_heatmap.png)")
log()
log("![표본외 자산곡선](oos_equity.png)")
(OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
r1.to_csv(OUT / "stage1_runs.csv", index=False)
r2.to_csv(OUT / "stage2_runs.csv", index=False)
(OUT / "best_params.json").write_text(json.dumps({k: (v.item() if hasattr(v, "item") else v)
                                                  for k, v in best2.items()}, ensure_ascii=False, indent=2))
print("\nsaved →", OUT)
