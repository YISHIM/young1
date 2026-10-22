"""후속 검증 3종
 A. 다중 이평선 앙상블: 여러 이평선 중 '보유' 신호 비율만큼 투자 (분산 효과가 진짜 원천인가)
 B. 기울기 국면별 '보유 비중' 조절: 기울기 정보가 낙폭 방어에 쓸모 있는가 (위약 검정 포함)
 C. 투자대상 확대: 1999년 말 기준 대형주 50종목(이후 상장폐지 종목 포함)으로 같은 검증 반복

실행:  python run_ensemble_experiment.py      (결과: results/ensemble/)
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from trend_lab import regime as rg
from trend_lab.backtest import Backtester, metrics

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "ensemble"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261023)
warnings.filterwarnings("ignore", category=RuntimeWarning)  # 상장폐지 종목의 빈 구간

IS = ("2000-01-01", "2015-12-31")
OOS = ("2016-01-01", "2026-12-31")
FULL = ("2000-01-01", "2026-12-31")
IS_PERIODS = [("2000-01-01", "2005-12-31"), ("2006-01-01", "2010-12-31"), ("2011-01-01", "2015-12-31")]
BASE = {"ma_type": "EMA", "buy_pct": 1.5, "sell_pct": 4.0, "trend_len": 225}  # 지난 다구치 최적
LENS = list(range(10, 101, 5)) + [120, 150, 200]
E4 = [20, 50, 100, 200]
LOOKBACKS, SCHEMES = [20, 60, 120], ["Q3", "Z2", "Z3"]
EXPO = [0.0, 0.25, 0.5, 0.75, 1.0]
N_BOOT, N_PLACEBO_B, N_PLACEBO_C = 2000, 1000, 200
BLOCK = 63

report: list[str] = []


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


# ------------------------------------------------------------------ 데이터
etf = pd.read_csv(ROOT / "data/etf_adj_close.csv", index_col=0, parse_dates=True)
stk = pd.read_csv(ROOT / "data/stocks50_adj_close.csv", index_col=0, parse_dates=True).reindex(etf.index)
rf = pd.read_csv(ROOT / "data/risk_free.csv", index_col=0, parse_dates=True)["rf_annual_pct"]
rf_d = (1 + rf / 100) ** (1 / 252) - 1
spy = etf["SPY"]
idx = etf.index
T = len(idx)
U = {"ETF14": Backtester(etf, rf_d, bench=spy), "STOCK50": Backtester(stk, rf_d, bench=spy)}
U_KO = {"ETF14": "ETF 14종", "STOCK50": "대형주 50종목(1999 기준)"}
POS = {u: {L: b.signal(BASE["ma_type"], L, BASE["buy_pct"], BASE["sell_pct"], BASE["trend_len"]) for L in LENS}
       for u, b in U.items()}
POS_NOF = {u: {L: b.signal(BASE["ma_type"], L, BASE["buy_pct"], BASE["sell_pct"], 0) for L in E4} for u, b in U.items()}


def mask(span):
    return (idx >= span[0]) & (idx <= span[1])


def port(r: np.ndarray) -> np.ndarray:
    """동일가중 포트폴리오 일간수익 (상장 중인 종목만, 상장폐지 시 남은 종목으로 재배분)."""
    out = np.full(len(r), np.nan)
    ok = ~np.all(np.isnan(r), axis=1)
    out[ok] = np.nanmean(r[ok], axis=1)
    return out


def summarize(b: Backtester, r: np.ndarray, pos: np.ndarray | None, span) -> dict:
    m = mask(span)
    mm = metrics(r[m], b.rf[m])
    pr = port(r)[m]
    w = np.cumprod(1 + np.nan_to_num(pr))
    ex = pr - b.rf[m]
    out = {"종목평균 CAGR": np.nanmean(mm["cagr"]), "종목평균 샤프": np.nanmean(mm["sharpe"]),
           "종목평균 MDD": np.nanmean(mm["mdd"]), "포트 CAGR": w[-1] ** (252 / len(pr)) - 1,
           "포트 샤프": np.nanmean(ex) / np.nanstd(ex, ddof=1) * np.sqrt(252),
           "포트 MDD": (w / np.maximum.accumulate(w) - 1).min()}
    if pos is not None:
        live = ~np.isnan(b.px[m])
        out["평균 투자비중"] = pos[m][live].mean()
        out["연 매매회전"] = np.abs(np.diff(pos[m], axis=0))[live[1:]].sum() / live.sum() * 252
    return out


def sharpe(x):
    x = x[~np.isnan(x)]
    return x.mean() / x.std(ddof=1) * np.sqrt(252)


def block_boot_diff(a: np.ndarray, b: np.ndarray, n=N_BOOT, block=BLOCK):
    """두 전략 포트폴리오 초과수익의 샤프 차이에 대한 짝지은 블록 부트스트랩 → (차이, 90% 구간, 단측 p)."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    L = len(a)
    nb = L // block + 1
    d = np.empty(n)
    for i in range(n):
        st = RNG.integers(0, L - block, nb)
        ix = (st[:, None] + np.arange(block)).ravel()[:L]
        d[i] = sharpe(a[ix]) - sharpe(b[ix])
    return sharpe(a) - sharpe(b), np.percentile(d, [5, 95]), (d <= 0).mean()


log("# 후속 검증 리포트 — 앙상블 · 기울기별 비중 조절 · 투자대상 확대")
log()
log("- 공통 매매 규칙: EMA, 이평선 +1.5% 돌파 매수 / −4% 이탈 매도, SPY>SMA225 장기추세 필터 (지난 다구치 최적)")
log("- 다음날 종가 체결, 편도 0.1% 비용, 현금은 단기국채 이자. 학습 2000–2015 / 표본외 2016–2026")
log(f"- 투자대상 ① ETF 14종  ② **대형주 50종목**: 1999년 말 S&P500 편입 종목 중 1999년 하루평균 거래대금 상위 50개. "
    f"당시 시점 정보로만 골랐고 이후 상장폐지·피인수 종목(Time Warner, Yahoo, Dell, EMC, 옛 DuPont, Xilinx 등) 포함 → 생존자 편향 없음")
log()

# ================================================================== A. 앙상블
log("## A. 다중 이평선 앙상블")
log()
log(f"**규칙**: 이평선 {E4}일 각각의 매수/매도 신호를 따로 계산하고, '보유' 신호를 낸 이평선 비율만큼 투자 "
    "(4개 중 3개 보유 → 75%). 사전에 정한 **주 가설**: 앙상블 E4가 단일 EMA100(지난 최적)보다 샤프가 높다.")
log()
ens_ret, ens_pos = {}, {}
for u, b in U.items():
    cands = {}
    for L in E4:
        cands[f"단일 EMA{L}"] = POS[u][L]
    cands["앙상블 E4 (20·50·100·200)"] = np.mean([POS[u][L] for L in E4], axis=0)
    cands["앙상블 전체 22개(10~200)"] = np.mean([POS[u][L] for L in LENS], axis=0)
    cands["앙상블 E4, 장기필터 없음"] = np.mean([POS_NOF[u][L] for L in E4], axis=0)
    ens_pos[u] = cands
    ens_ret[u] = {n: b.returns(p) for n, p in cands.items()}
    ens_ret[u]["매수 후 보유"] = b.buy_and_hold()
    ens_pos[u]["매수 후 보유"] = np.ones_like(b.px)
    for label, span in [("학습 2000–2015", IS), ("표본외 2016–2026", OOS), ("전체 2000–2026", FULL)]:
        t = pd.DataFrame({n: summarize(b, r, ens_pos[u][n], span) for n, r in ens_ret[u].items()}).T
        t.index.name = f"{U_KO[u]} · {label}"
        log(md_table(t))
        log()

log("### 통계 검정 — 앙상블 E4 vs 단일 EMA100 (포트폴리오 샤프 차이, 63일 블록 부트스트랩 2,000회)")
log()
rows = []
for u, b in U.items():
    a_r = port(ens_ret[u]["앙상블 E4 (20·50·100·200)"]) - b.rf
    s_r = port(ens_ret[u]["단일 EMA100"]) - b.rf
    for label, span in [("학습", IS), ("표본외", OOS), ("전체", FULL)]:
        m = mask(span)
        d, ci, p = block_boot_diff(a_r[m], s_r[m])
        # 종목별 승률
        ma = metrics(ens_ret[u]["앙상블 E4 (20·50·100·200)"][m], b.rf[m])["sharpe"]
        ms = metrics(ens_ret[u]["단일 EMA100"][m], b.rf[m])["sharpe"]
        ok = ~np.isnan(ma) & ~np.isnan(ms)
        wins = int((ma[ok] > ms[ok]).sum())
        rows.append({"투자대상·구간": f"{U_KO[u]} · {label}", "샤프 차이": d, "90% 구간": f"{ci[0]:+.3f} ~ {ci[1]:+.3f}",
                     "p값(단측)": p, "종목별 승": f"{wins}/{ok.sum()}",
                     "부호검정 p": binomtest(wins, int(ok.sum()), 0.5, alternative="greater").pvalue})
bt_tab = pd.DataFrame(rows).set_index("투자대상·구간")
log(md_table(bt_tab))
log()

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
for ax, (u, b) in zip(axes, U.items()):
    m = mask(FULL)
    for n, en in [("앙상블 E4 (20·50·100·200)", "Ensemble E4"), ("단일 EMA100", "Single EMA100"),
                  ("단일 EMA20", "Single EMA20"), ("매수 후 보유", "Buy & Hold")]:
        ax.plot(idx[m], np.cumprod(1 + np.nan_to_num(port(ens_ret[u][n])[m])), label=en)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"{u}: equal-weight portfolio 2000-2026")
fig.tight_layout()
fig.savefig(OUT / "A_ensemble_equity.png", dpi=110)
plt.close(fig)

# ================================================================== B. 기울기 국면별 보유 비중
log("## B. 기울기 국면별 보유 비중 조절 — 기울기 정보가 낙폭 방어에 쓸모 있는가")
log()
log("**규칙**: 기본 신호(단일 EMA100 + SMA225 필터)가 '보유'일 때, SPY 200일선 기울기 국면에 따라 보유 비중을 "
    f"{EXPO} 중에서 국면별로 다르게 둔다(최대 비중 = 1로 정규화). 설정 9개(기울기 기간 {LOOKBACKS} × 국면구분 Q3/Z2/Z3).")
log()
log("비중을 줄이면 낙폭은 당연히 줄어든다. 그래서 **같은 평균 비중을 항상 일정하게 유지하는 전략**과 비교한다. "
    "샤프는 일정 비중으로는 바뀌지 않으므로, 샤프 개선 = 순수하게 '국면 판단'의 효과다.")
log()
SLOPES = {k: rg.slope(spy, 200, k) for k in LOOKBACKS}
ISM = mask(IS)


def expo_search(u: str, g_shift: int = 0, span=IS):
    """설정 9개 × 비중 조합 전수조사 (학습구간 포트 샤프 최대화). 비용 변화는 무시한 근사."""
    b = U[u]
    base_ex = port(b.returns(POS[u][100])) - b.rf
    m = mask(span)
    best = None
    out = {}
    for k in LOOKBACKS:
        for s in SCHEMES:
            cuts = rg.thresholds(SLOPES[k], s, ISM)
            g = rg.classify(SLOPES[k], cuts)
            if g_shift:
                g = np.roll(g, g_shift)
            gl = np.full_like(g, g[0])
            gl[2:] = g[:-2]
            K = len(cuts) + 1
            x = base_ex[m]
            gg = gl[m]
            ok = ~np.isnan(x)
            S1 = np.array([x[ok & (gg == j)].sum() for j in range(K)])
            S2 = np.array([(x[ok & (gg == j)] ** 2).sum() for j in range(K)])
            N = ok.sum()
            combos = rg.all_combos(len(EXPO), K)
            e = np.array(EXPO)[combos]
            e = e[e.max(1) == 1.0]
            mu = e @ S1 / N
            var = (e ** 2 @ S2 / N - mu ** 2) * N / (N - 1)
            sh = mu / np.sqrt(var) * np.sqrt(252)
            i = int(np.nanargmax(sh))
            base_sh = S1.sum() / N / np.sqrt((S2.sum() / N - (S1.sum() / N) ** 2) * N / (N - 1)) * np.sqrt(252)
            out[(k, s)] = {"e": tuple(float(v) for v in e[i]), "sharpe": sh[i], "base": base_sh, "delta": sh[i] - base_sh,
                           "g": g, "cuts": cuts}
            if best is None or out[(k, s)]["delta"] > out[best]["delta"]:
                best = (k, s)
    return out, best


expo_res, expo_best = expo_search("ETF14")
rows = []
for (k, s), r in expo_res.items():
    rows.append({"기울기기간": k, "국면구분": s, "국면별 비중(약→강)": " / ".join(f"{v:.2f}" for v in r["e"]),
                 "최적 샤프": r["sharpe"], "기본 샤프": r["base"], "Δ샤프": r["delta"]})
log("ETF 14종, 학습구간 2000–2015 전수조사:")
log()
log(md_table(pd.DataFrame(rows).set_index("기울기기간")))
log()
t0 = time.time()
real_d = expo_res[expo_best]["delta"]
plc_max = np.array([max(r["delta"] for r in expo_search("ETF14", int(RNG.integers(252, T - 252)))[0].values())
                    for _ in range(N_PLACEBO_B)])
p_b = (plc_max >= real_d).mean()
print(f"placebo B {time.time() - t0:.0f}s")
log(f"- **위약 검정 {N_PLACEBO_B}회** (국면 라벨 무작위 이동, 9개 설정 중 최댓값 비교): 실제 최대 Δ샤프 {real_d:.3f}, "
    f"위약 평균 {plc_max.mean():.3f}, 위약 95% {np.percentile(plc_max, 95):.3f} → **p = {p_b:.3f}**")
log()

# 최고 설정을 정확한 백테스트로: 학습/표본외/종목 확대(학습 설정 그대로 적용)
k_b, s_b = expo_best
eb = expo_res[expo_best]
log(f"최고 설정: 기울기 {k_b}일, {s_b}, 국면별 비중 {eb['e']}. 이 설정을 고정한 채 **표본외 구간**과 **대형주 50종목**에 그대로 적용:")
log()
rows = []
for u, b in U.items():
    g = eb["g"]
    base_pos = POS[u][100]
    scaled = base_pos * np.array(eb["e"])[g][:, None]
    r_sc = b.returns(scaled)
    r_base = b.returns(base_pos)
    for label, span in [("학습", IS), ("표본외", OOS), ("전체", FULL)]:
        m = mask(span)
        avg_e = scaled[m][~np.isnan(b.px[m])].sum() / max(base_pos[m][~np.isnan(b.px[m])].sum(), 1)
        r_const = b.returns(base_pos * avg_e)  # 같은 평균 비중을 항상 유지
        s_sc, s_c, s_b0 = (summarize(b, r_sc, None, span), summarize(b, r_const, None, span),
                           summarize(b, r_base, None, span))
        d, ci, p = block_boot_diff((port(r_sc) - b.rf)[m], (port(r_base) - b.rf)[m])
        rows.append({"투자대상·구간": f"{U_KO[u]} · {label}", "기본 샤프": s_b0["포트 샤프"], "국면비중 샤프": s_sc["포트 샤프"],
                     "샤프차 p(부트)": p, "기본 MDD": s_b0["포트 MDD"], "국면비중 MDD": s_sc["포트 MDD"],
                     f"일정비중 MDD": s_c["포트 MDD"], "국면비중 CAGR": s_sc["포트 CAGR"], "일정비중 CAGR": s_c["포트 CAGR"]})
log(md_table(pd.DataFrame(rows).set_index("투자대상·구간")))
log()
log("※ '일정비중' = 국면비중 전략과 평균 투자비중이 같도록 항상 고정 비율만 보유. "
    "국면비중의 MDD·CAGR이 일정비중보다 좋아야 기울기 판단이 가치가 있다.")
log()
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.hist(plc_max, bins=40, color="#94a3b8")
ax.axvline(real_d, color="#dc2626", lw=2, label=f"real = {real_d:.3f}")
ax.set_title(f"B: exposure-by-slope placebo (n={N_PLACEBO_B}), p = {p_b:.3f}")
ax.set_xlabel("Max Sharpe improvement over base (9 settings)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "B_placebo.png", dpi=110)
plt.close(fig)

# ================================================================== C. 대형주 50종목에서 '기울기별 이평선 전환' 재검증
log("## C. 대형주 50종목에서 '기울기별 이평선 전환' 가설 재검증")
log()
log("ETF에서 기각된 가설을 독립된 표본(개별 종목)에서 다시 시험: ① ETF에서 고른 최적 규칙(SMA225 필터, 60일 기울기, "
    "하락/완만/가파른 국면, 이평 10/25/120)을 그대로 적용  ② 종목 데이터로 18개 설정 전수조사 + 위약 검정.")
log()
FILTERS = [225, 0]
EXC = {}
for f in FILTERS:
    b = U["STOCK50"]
    EXC[f] = np.stack([b.returns(b.signal("EMA", L, 1.5, 4.0, f)) - b.rf[:, None] for L in LENS])
cells_all = np.full(T, -1)
for i, (a, z) in enumerate(IS_PERIODS):
    cells_all[(idx >= a) & (idx <= z)] = i
rows_is = np.where(cells_all >= 0)[0]


def search_c(f, k, scheme, g_shift=0):
    cuts = rg.thresholds(SLOPES[k], scheme, ISM)
    g = rg.classify(SLOPES[k], cuts)
    if g_shift:
        g = np.roll(g, g_shift)
    gl = np.full_like(g, g[0])
    gl[2:] = g[:-2]
    K = len(cuts) + 1
    ev = rg.FastSwitchEvaluator(EXC[f][:, rows_is], cells_all[rows_is], 3)
    n, s1, s2 = ev.sums(gl[rows_is], K)
    rob, _ = ev.score(n, s1, s2, rg.all_combos(len(LENS), K))
    fx, _ = ev.score(n, s1, s2, np.array([[i] * K for i in range(len(LENS))]))
    bi = int(np.nanargmax(rob))
    return {"choice": tuple(rg.all_combos(len(LENS), K)[bi]), "robust": rob[bi],
            "fixed": LENS[int(np.nanargmax(fx))], "fixed_robust": np.nanmax(fx), "delta": rob[bi] - np.nanmax(fx), "g": g}


SET_C = [(f, k, s) for f in FILTERS for k in LOOKBACKS for s in SCHEMES]
t0 = time.time()
res_c = {st: search_c(*st) for st in SET_C}
print(f"C search {time.time() - t0:.0f}s")
rows = [{"필터": f"SMA{f}" if f else "없음", "기울기기간": k, "국면구분": s,
         "국면별 이평": " / ".join(str(LENS[i]) for i in r["choice"]), "전환 점수": r["robust"],
         "단일 최적": r["fixed"], "단일 점수": r["fixed_robust"], "Δ": r["delta"]} for (f, k, s), r in res_c.items()]
tc = pd.DataFrame(rows).sort_values("전환 점수", ascending=False).reset_index(drop=True)
tc.index += 1
tc.index.name = "순위"
log("대형주 50종목, 학습구간 2000–2015 전수조사 (점수 = 3시기 × 50종목 = 150셀 샤프의 평균 − 표준편차):")
log()
log(md_table(tc))
log()
t0 = time.time()
real_c = max(r["delta"] for r in res_c.values())
plc_c = []
for i in range(N_PLACEBO_C):
    sh = int(RNG.integers(252, T - 252))
    plc_c.append(max(search_c(*st, g_shift=sh)["delta"] for st in SET_C))
    if i % 20 == 0:
        print(f"  placebo C {i}/{N_PLACEBO_C} {time.time() - t0:.0f}s", flush=True)
plc_c = np.array(plc_c)
p_c = (plc_c >= real_c).mean()
log(f"- **위약 검정 {N_PLACEBO_C}회**: 실제 최대 Δ {real_c:.3f}, 위약 평균 {plc_c.mean():.3f}, "
    f"위약 95% {np.percentile(plc_c, 95):.3f} → **p = {p_c:.3f}**")
log()
# ETF에서 고른 규칙을 종목에 그대로 적용 (표본외 성격의 재현 시험)
b = U["STOCK50"]
cuts = rg.thresholds(SLOPES[60], "Z3", ISM)
g = rg.classify(SLOPES[60], cuts)
etf_rule = rg.combine([POS["STOCK50"][L] for L in (10, 25, 120)], (0, 1, 2), g)
comp = {"ETF 최적 전환규칙(10/25/120)": (b.returns(etf_rule), etf_rule),
        "단일 EMA120": (b.returns(POS["STOCK50"][120]), POS["STOCK50"][120]),
        "단일 EMA100(다구치)": (ens_ret["STOCK50"]["단일 EMA100"], POS["STOCK50"][100]),
        "앙상블 E4": (ens_ret["STOCK50"]["앙상블 E4 (20·50·100·200)"], ens_pos["STOCK50"]["앙상블 E4 (20·50·100·200)"]),
        "매수 후 보유": (b.buy_and_hold(), np.ones_like(b.px))}
for label, span in [("전체 2000–2026", FULL), ("표본외 2016–2026", OOS)]:
    t = pd.DataFrame({n: summarize(b, r, p, span) for n, (r, p) in comp.items()}).T
    t.index.name = f"대형주 50종목 · {label}"
    log(md_table(t))
    log()

log("![A 앙상블](A_ensemble_equity.png)")
log()
log("![B 위약](B_placebo.png)")
(OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
print("saved →", OUT)
