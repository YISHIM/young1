"""가설 검증: '장기추세의 기울기에 따라 최적 이평선이 다르다' → 기울기 국면별 이평선 전환 규칙.

실행:  python run_slope_experiment.py      (결과: results/slope/)

검증 단계
 1. 국면 정의(기울기 측정 기간 3종 × 국면 나누는 방식 3종) × 장기추세 필터 유무 = 18개 설정
 2. 진단: 국면별·이평선별 초과수익 지도 + 블록 부트스트랩으로 '국면별 최적 이평선'의 신뢰구간
 3. 설정마다 국면별 이평선 조합 전수조사(최대 22³ = 10,648개) → 단일 이평선 최적과 비교
 4. 위약(placebo) 검정: 국면 라벨을 시간축에서 무작위로 밀어 '기울기 정보'만 없앤 채 같은 절차 반복
 5. 학습구간 내 국면 3개 시기별 일관성
 6. 정확한 백테스트로 학습/표본외(2016~) 성과 확인
 7. 워크포워드: 매년 초 직전 데이터만으로 설정·조합을 다시 고르고 다음 1년 매매 (2006~2026)
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trend_lab import regime as rg
from trend_lab.backtest import Backtester, metrics

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "slope"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261022)

UNIVERSE = ["SPY", "QQQ", "DIA", "IWM", "MDY",
            "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
IS_PERIODS = [("2000-01-01", "2005-12-31"), ("2006-01-01", "2010-12-31"), ("2011-01-01", "2015-12-31")]
OOS = ("2016-01-01", "2026-12-31")
LENS = list(range(10, 101, 5)) + [120, 150, 200]           # 후보 이평선 22개
BASE = {"ma_type": "EMA", "buy_pct": 1.5, "sell_pct": 4.0}  # 지난 다구치 최적에서 고정
FILTERS = [225, 0]                                         # 장기추세 필터: SPY>SMA225 / 없음
SLOPE_MA = 200                                             # 기울기를 잴 장기 이평선
LOOKBACKS = [20, 60, 120]                                  # 기울기 측정 기간(일)
SCHEMES = ["Q3", "Z2", "Z3"]
SCHEME_KO = {"Q3": "3분위(하/중/상)", "Z2": "하락/상승(0 기준)", "Z3": "하락/완만상승/가파른상승"}
N_PLACEBO = 500
N_BOOT = 1000
N_WF_PLACEBO = 50

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


def lens_str(choice) -> str:
    return " / ".join(str(LENS[i]) for i in choice)


# ------------------------------------------------------------------ 준비
price = pd.read_csv(ROOT / "data/etf_adj_close.csv", index_col=0, parse_dates=True)[UNIVERSE]
rf = pd.read_csv(ROOT / "data/risk_free.csv", index_col=0, parse_dates=True)["rf_annual_pct"]
bt = Backtester(price, (1 + rf / 100) ** (1 / 252) - 1, bench="SPY", cost=0.001)
idx = price.index
T = len(idx)
t0 = time.time()
POS = {f: [bt.signal(BASE["ma_type"], L, BASE["buy_pct"], BASE["sell_pct"], f) for L in LENS] for f in FILTERS}
EX = {f: np.stack([bt.returns(p) - bt.rf[:, None] for p in POS[f]]) for f in FILTERS}  # [L,T,J]
SLOPES = {k: rg.slope(price["SPY"], SLOPE_MA, k) for k in LOOKBACKS}
SETTINGS = [(f, k, s) for f in FILTERS for k in LOOKBACKS for s in SCHEMES]
print(f"precompute {time.time() - t0:.1f}s")


def period_cells(periods) -> np.ndarray:
    c = np.full(T, -1)
    for i, (a, b) in enumerate(periods):
        c[(idx >= a) & (idx <= b)] = i
    return c


def lag2(g: np.ndarray) -> np.ndarray:
    """신호일 국면 → 수익 발생일 국면 (다음날 종가 체결, 그다음 날부터 수익)."""
    out = np.full_like(g, g[0])
    out[2:] = g[:-2]
    return out


def search(f, k, scheme, train_mask, periods, g_shift=0):
    """한 설정에서 학습 구간 기준 최적 국면 조합 탐색. g_shift≠0 이면 위약(라벨 순환 이동)."""
    s = SLOPES[k]
    cuts = rg.thresholds(s, scheme, train_mask)
    g = rg.classify(s, cuts)
    if g_shift:
        g = np.roll(g, g_shift)
    K = len(cuts) + 1
    cells = period_cells(periods)
    rows = np.where(cells >= 0)[0]
    ev = rg.FastSwitchEvaluator(EX[f][:, rows], cells[rows], len(periods))
    n, s1, s2 = ev.sums(lag2(g)[rows], K)
    combos = rg.all_combos(len(LENS), K)
    robust, mean = ev.score(n, s1, s2, combos)
    fixed = np.array([[i] * K for i in range(len(LENS))])
    rf_, mf_ = ev.score(n, s1, s2, fixed)
    b, bf = int(np.nanargmax(robust)), int(np.nanargmax(rf_))
    return {"cuts": cuts, "g": g, "K": K, "choice": tuple(combos[b]), "robust": robust[b], "sharpe": mean[b],
            "fixed_len": bf, "fixed_robust": rf_[bf], "fixed_sharpe": mf_[bf],
            "delta": robust[b] - rf_[bf], "_ev": (n, s1, s2)}


IS_MASK = (idx >= IS_PERIODS[0][0]) & (idx <= IS_PERIODS[-1][1])

log("# 장기추세 기울기별 이평선 전환 규칙 — 철저 검증 리포트")
log()
log("**가설**: 장기추세(SPY 200일선)의 기울기 국면마다 최적 이평선이 다르다. "
    "그렇다면 국면에 따라 이평선을 바꿔 쓰는 규칙이 하나의 이평선을 고정해 쓰는 것보다 낫다.")
log()
log(f"- 규칙: t일 종가에 SPY {SLOPE_MA}일선의 기울기 = (MA_t / MA_(t−k) − 1)×252/k 로 국면 판정 → "
    f"그 국면에 배정된 이평선의 매수/매도 신호를 따름 (매수 +{BASE['buy_pct']}% 돌파, 매도 −{BASE['sell_pct']}% 이탈, {BASE['ma_type']})")
log(f"- 후보 이평선 {len(LENS)}개: {LENS}")
log(f"- 설정 18개 = 기울기 측정기간 {LOOKBACKS}일 × 국면 구분 {list(SCHEME_KO.values())} × 장기추세 필터(SPY>SMA225 / 없음)")
log("- 국면 경계값은 학습구간(2000–2015) 데이터로만 결정. 평가 점수는 지난 실험과 동일: "
    "3국면 × 14종목 = 42개 셀의 샤프 **평균 − 표준편차** (로버스트 점수)")
log("- 데이터·비용·체결 가정은 지난 실험과 동일 (다음날 종가 체결, 편도 0.1%, 현금은 단기국채 이자)")
log()

# ------------------------------------------------------------------ 1. 국면 설명
log("## 1. 국면은 어떤 시장이었나 (기울기 측정 60일, 3분위 기준, 학습구간)")
log()
s60 = SLOPES[60]
cuts = rg.thresholds(s60, "Q3", IS_MASK)
g60 = rg.classify(s60, cuts)
spy_r = price["SPY"].pct_change().to_numpy()
fwd = price["SPY"].shift(-63).to_numpy() / price["SPY"].to_numpy() - 1
desc = []
for k, name in enumerate(["하위(약세)", "중위", "상위(강세)"]):
    m = IS_MASK & (g60 == k)
    desc.append({"국면": name, "기울기 범위(연율)": ["< %.1f%%" % (cuts[0] * 100),
                                                f"{cuts[0] * 100:.1f}% ~ {cuts[1] * 100:.1f}%",
                                                "> %.1f%%" % (cuts[1] * 100)][k],
                 "일수 비중": m.sum() / IS_MASK.sum(),
                 "SPY 연율 수익": np.nanmean(spy_r[m]) * 252, "SPY 연율 변동성": np.nanstd(spy_r[m]) * np.sqrt(252),
                 "이후 3개월 SPY 수익(평균)": np.nanmean(fwd[m])})
log(md_table(pd.DataFrame(desc).set_index("국면")))
log()

# ------------------------------------------------------------------ 2. 진단 지도 + 부트스트랩
log("## 2. 진단 — 국면별로 '어떤 이평선이 돈을 벌었나' (학습구간, 필터 없음)")
log()
log("각 이평선 전략을 그대로 돌린 뒤, 국면별 날짜만 모아 14종 평균 **연율 초과수익(무위험 대비)**을 계산. "
    "블록 부트스트랩(63일 블록, 1,000회)으로 국면별 1등 이평선이 얼마나 확실한지 측정.")
log()
ex_ew = np.nanmean(EX[0], axis=2)  # [L,T] 14종 동일가중 평균 초과수익
gl = lag2(g60)
is_rows = np.where(IS_MASK)[0]
heat = np.array([[np.nanmean(ex_ew[li, is_rows][gl[is_rows] == k]) * 252 for li in range(len(LENS))] for k in range(3)])
# 블록 부트스트랩
B = 63
nb = len(is_rows) // B + 1
win = np.zeros((3, len(LENS)))
for _ in range(N_BOOT):
    starts = RNG.integers(0, len(is_rows) - B, nb)
    rows = is_rows[(starts[:, None] + np.arange(B)).ravel()[:len(is_rows)]]
    e, gg = ex_ew[:, rows], gl[rows]
    for k in range(3):
        mk = gg == k
        win[k, np.argmax(np.nanmean(e[:, mk], axis=1))] += 1
win /= N_BOOT
diag = []
for k, name in enumerate(["하위(약세)", "중위", "상위(강세)"]):
    top3 = np.argsort(-win[k])[:3]
    lo, hi = np.percentile(np.repeat(LENS, (win[k] * N_BOOT).astype(int)), [5, 95])
    diag.append({"국면": name, "1등 이평선": LENS[int(np.argmax(heat[k]))],
                 "1등 초과수익(연율)": heat[k].max(), "꼴찌 초과수익": heat[k].min(),
                 "부트스트랩 1등 빈도 Top3": ", ".join(f"{LENS[i]}일 {win[k, i]:.0%}" for i in top3),
                 "1등 이평선 90% 구간": f"{int(lo)}–{int(hi)}일"})
log(md_table(pd.DataFrame(diag).set_index("국면")))
log()
fig, ax = plt.subplots(figsize=(13, 3.2))
im = ax.imshow(heat * 100, aspect="auto", cmap="RdYlGn")
ax.set_xticks(range(len(LENS)), LENS)
ax.set_yticks(range(3), ["Low slope", "Mid slope", "High slope"])
for k in range(3):
    ax.plot(np.argmax(heat[k]), k, "k*", ms=12)
ax.set_xlabel("MA length (days)")
ax.set_title("Annualized excess return by slope regime (in-sample 2000-2015, no filter, 14-ETF avg, %)  * = best")
fig.colorbar(im)
fig.tight_layout()
fig.savefig(OUT / "regime_ma_heatmap.png", dpi=110)
plt.close(fig)

# ------------------------------------------------------------------ 3. 설정별 전수조사
log("## 3. 설정 18개 × 국면별 이평선 조합 전수조사 (학습구간 2000–2015)")
log()
t0 = time.time()
RES = {st: search(*st, IS_MASK, IS_PERIODS) for st in SETTINGS}
rows = []
for (f, k, s), r in RES.items():
    rows.append({"필터": f"SMA{f}" if f else "없음", "기울기기간": k, "국면구분": SCHEME_KO[s],
                 "국면별 이평(약→강)": lens_str(r["choice"]), "전환 점수": r["robust"],
                 "단일 최적": LENS[r["fixed_len"]], "단일 점수": r["fixed_robust"], "개선폭 Δ": r["delta"]})
tab = pd.DataFrame(rows).sort_values("전환 점수", ascending=False).reset_index(drop=True)
tab.index = tab.index + 1
tab.index.name = "순위"
log(md_table(tab))
log()
log(f"(전수조사 평가 {sum(len(LENS) ** r['K'] for r in RES.values()):,}개 조합, {time.time() - t0:.1f}초)")
log()
BEST = max(RES, key=lambda st: RES[st]["robust"])
br = RES[BEST]
log(f"→ 최고 설정: 필터 {'SMA' + str(BEST[0]) if BEST[0] else '없음'}, 기울기 {BEST[1]}일, {SCHEME_KO[BEST[2]]}, "
    f"국면별 이평 **{lens_str(br['choice'])}일** (로버스트 점수 {br['robust']:.3f} vs 단일 {LENS[br['fixed_len']]}일 {br['fixed_robust']:.3f})")
log()

# ------------------------------------------------------------------ 4. 위약 검정
log("## 4. 위약(placebo) 검정 — 개선이 '기울기 정보' 덕분인가, 단지 '선택지가 늘어서'인가")
log()
log("전환 규칙은 단일 이평선을 특수한 경우로 포함하므로 학습구간에서는 **항상** 같거나 더 좋게 나온다. "
    "그래서 국면 라벨을 시간축에서 무작위로 1년 이상 밀어(국면의 길이·비율은 유지, 기울기와의 연결만 끊음) "
    f"18개 설정 전체 탐색을 {N_PLACEBO}회 반복했다. 실제 Δ가 가짜 Δ보다 충분히 커야 기울기 정보가 진짜다.")
log()
t0 = time.time()
plc = np.zeros((N_PLACEBO, len(SETTINGS)))
for i in range(N_PLACEBO):
    sh = int(RNG.integers(252, T - 252))
    for j, st in enumerate(SETTINGS):
        plc[i, j] = search(*st, IS_MASK, IS_PERIODS, g_shift=sh)["delta"]
    if i % 100 == 0:
        print(f"  placebo {i}/{N_PLACEBO} {time.time() - t0:.0f}s", flush=True)
real = np.array([RES[st]["delta"] for st in SETTINGS])
p_each = (plc >= real[None]).mean(0)
p_max = (plc.max(1) >= real.max()).mean()
pt = pd.DataFrame({"필터": [f"SMA{f}" if f else "없음" for f, _, _ in SETTINGS],
                   "기울기기간": [k for _, k, _ in SETTINGS], "국면구분": [SCHEME_KO[s] for *_, s in SETTINGS],
                   "실제 Δ": real, "위약 Δ 평균": plc.mean(0), "위약 Δ 95%": np.percentile(plc, 95, axis=0),
                   "p값": p_each})
pt.index = range(1, len(pt) + 1)
pt.index.name = "#"
log(md_table(pt))
log()
log(f"- **전체 절차 기준(18개 중 최댓값 비교) p값 = {p_max:.3f}** — 여러 설정을 시험한 것까지 감안한 공정한 p값")
log(f"- 개별 p<0.05 설정 수: {(p_each < 0.05).sum()}/18 (우연이라도 약 1개는 나올 수 있음)")
log()
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.hist(plc.max(1), bins=40, color="#94a3b8")
ax.axvline(real.max(), color="#dc2626", lw=2, label=f"real max delta = {real.max():.3f}")
ax.set_xlabel("Max improvement over single-MA (18 settings), placebo")
ax.set_title(f"Placebo test (n={N_PLACEBO}), p = {p_max:.3f}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "placebo.png", dpi=110)
plt.close(fig)

# ------------------------------------------------------------------ 5. 시기별 일관성
log("## 5. 시기별 일관성 — 학습구간 3개 시기에서 따로 최적화하면 같은 답이 나오나")
log()
cons = []
for a, b in IS_PERIODS:
    m = (idx >= a) & (idx <= b)
    r = search(*BEST, IS_MASK, [(a, b)])  # 경계값은 전체 학습구간 기준 고정
    cons.append({"시기": f"{a[:4]}–{b[:4]}", "국면별 최적 이평(약→강)": lens_str(r["choice"]),
                 "단일 최적": LENS[r["fixed_len"]], "Δ": r["delta"]})
log(md_table(pd.DataFrame(cons).set_index("시기")))
log()

# ------------------------------------------------------------------ 6. 정확한 백테스트 (학습/표본외)
log("## 6. 정확한 백테스트 — 전환 시점 매매비용까지 반영, 학습구간 vs 표본외(2016–2026)")
log()
f_b = BEST[0]
sw_pos = rg.combine(POS[f_b], br["choice"], br["g"])
cands = {
    "기울기 전환 규칙(최고 설정)": bt.returns(sw_pos),
    f"단일 이평 {LENS[br['fixed_len']]}일(같은 설정의 단일 최적)": bt.returns(POS[f_b][br["fixed_len"]]),
    "지난 다구치 최적 EMA100/+1.5/−4/SMA225": bt.run("EMA", 100, 1.5, 4.0, 225),
    "매수 후 보유": bt.buy_and_hold(),
}


def summarize(r, a, b):
    m = (idx >= a) & (idx <= b)
    mm = metrics(r[m], bt.rf[m])
    pr = np.nanmean(r[m], axis=1)
    w = np.cumprod(1 + pr)
    ex = pr - bt.rf[m]
    return {"종목평균 CAGR": np.nanmean(mm["cagr"]), "종목평균 샤프": np.nanmean(mm["sharpe"]),
            "종목평균 MDD": np.nanmean(mm["mdd"]), "포트 CAGR": w[-1] ** (252 / len(pr)) - 1,
            "포트 샤프": ex.mean() / ex.std(ddof=1) * np.sqrt(252), "포트 MDD": (w / np.maximum.accumulate(w) - 1).min()}


for label, (a, b) in [("학습구간 2000–2015", ("2000-01-01", "2015-12-31")), ("표본외 2016–2026", OOS)]:
    t = pd.DataFrame({n: summarize(r, a, b) for n, r in cands.items()}).T
    t.index.name = label
    log(md_table(t))
    log()

# ------------------------------------------------------------------ 7. 워크포워드
log("## 7. 워크포워드 — 매년 초, 그 전까지의 데이터만으로 '설정 + 국면별 이평'을 다시 골라 1년 운용")
log()
log("비교 대상도 같은 방식으로 매년 다시 고른 **단일 이평선**(필터 유무 포함)이다. 둘 다 미래 정보 없음.")
log()
def walk_forward(g_shift: int = 0):
    wf_sw = np.full((T, len(UNIVERSE)), np.nan)
    wf_fx = np.full((T, len(UNIVERSE)), np.nan)
    wf_log = []
    for Y in range(2006, idx[-1].year + 1):
        train = (idx >= "2000-01-01") & (idx < f"{Y}-01-01")
        d = idx[train]
        edges = np.linspace(0, len(d) - 1, 4).astype(int)
        periods = [(str(d[edges[i]].date()), str(d[edges[i + 1] - (1 if i < 2 else 0)].date())) for i in range(3)]
        res = {st: search(*st, train, periods, g_shift=g_shift) for st in SETTINGS}
        st = max(res, key=lambda s: res[s]["robust"])
        r = res[st]
        # 단일 이평선 최적: 필터별 단일 최적 중 점수 높은 것
        fbest = max(FILTERS, key=lambda f: max(res[s]["fixed_robust"] for s in SETTINGS if s[0] == f))
        s_fx = next(s for s in SETTINGS if s[0] == fbest)
        li_fx = res[s_fx]["fixed_len"]
        test = (idx >= f"{Y}-01-01") & (idx <= f"{Y}-12-31")
        wf_sw[test] = bt.returns(rg.combine(POS[st[0]], r["choice"], r["g"]))[test]
        wf_fx[test] = bt.returns(POS[fbest][li_fx])[test]
        wf_log.append({"연도": Y, "선택 설정": f"{'SMA' + str(st[0]) if st[0] else '필터없음'}/{st[1]}일/{st[2]}",
                       "국면별 이평": lens_str(r["choice"]),
                       "단일 이평": f"{LENS[li_fx]}일({'SMA' + str(fbest) if fbest else '필터없음'})"})
    return wf_sw, wf_fx, wf_log


t0 = time.time()
wf_sw, wf_fx, wf_log = walk_forward()
print(f"walk-forward {time.time() - t0:.0f}s")
wl = pd.DataFrame(wf_log).set_index("연도")
log(md_table(wl))
log()
log("→ 해마다 고른 조합이 크게 흔들리면, 기울기별 최적 이평이 '안정된 법칙'이 아니라는 신호다.")
log()
wf = {"워크포워드 기울기 전환": wf_sw, "워크포워드 단일 이평": wf_fx,
      "지난 다구치 최적(고정)": cands["지난 다구치 최적 EMA100/+1.5/−4/SMA225"], "매수 후 보유": bt.buy_and_hold()}
for label, (a, b) in [("워크포워드 2006–2026", ("2006-01-01", "2026-12-31")), ("그중 2016–2026", OOS)]:
    t = pd.DataFrame({n: summarize(r, a, b) for n, r in wf.items()}).T
    t.index.name = label
    log(md_table(t))
    log()
# 연도별 승률 (포트폴리오 기준)
wm = idx >= "2006-01-01"
yr = pd.DataFrame({n: pd.Series(np.nanmean(r[wm], axis=1), index=idx[wm]) for n, r in wf.items()})
yr = (1 + yr).groupby(yr.index.year).prod() - 1
win_rate = (yr["워크포워드 기울기 전환"] > yr["워크포워드 단일 이평"]).mean()
diff = yr["워크포워드 기울기 전환"] - yr["워크포워드 단일 이평"]
tstat = diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff))
log(f"- 연도별로 기울기 전환이 단일 이평을 이긴 비율: **{win_rate:.0%}** ({len(yr)}년), "
    f"연평균 수익 차이 {diff.mean():+.2%} (t = {tstat:.2f}, |t|>2 이어야 통계적으로 의미)")
log()

# 워크포워드 위약: 국면 라벨만 무작위로 밀고 같은 워크포워드 절차 전체를 반복
WF_SPAN = ("2006-01-01", "2026-12-31")
real_wf = summarize(wf_sw, *WF_SPAN)
t0 = time.time()
plc_wf = []
for i in range(N_WF_PLACEBO):
    p_sw, _, _ = walk_forward(int(RNG.integers(252, T - 252)))
    plc_wf.append(summarize(p_sw, *WF_SPAN))
    print(f"  wf placebo {i + 1}/{N_WF_PLACEBO} {time.time() - t0:.0f}s", flush=True)
plc_wf = pd.DataFrame(plc_wf)
log(f"### 워크포워드 위약 검정 ({N_WF_PLACEBO}회) — 워크포워드 성과가 '기울기 정보' 덕분인가")
log()
log("국면 라벨만 시간축에서 무작위로 밀어(기울기와 무관한 가짜 국면) 위의 워크포워드 절차를 통째로 반복했다. "
    "가짜 국면으로도 비슷한 성과가 나오면, 개선은 기울기 덕분이 아니라 '여러 이평선을 섞어 쓰는 효과(분산)' 때문이다.")
log()
wp = pd.DataFrame({"실제(기울기 국면)": [real_wf["포트 샤프"], real_wf["포트 CAGR"], real_wf["포트 MDD"]],
                   "위약 평균": [plc_wf["포트 샤프"].mean(), plc_wf["포트 CAGR"].mean(), plc_wf["포트 MDD"].mean()],
                   "위약 95%": [plc_wf["포트 샤프"].quantile(.95), plc_wf["포트 CAGR"].quantile(.95),
                              plc_wf["포트 MDD"].quantile(.95)],
                   "p값(위약 ≥ 실제)": [(plc_wf["포트 샤프"] >= real_wf["포트 샤프"]).mean(),
                                    (plc_wf["포트 CAGR"] >= real_wf["포트 CAGR"]).mean(),
                                    (plc_wf["포트 MDD"] >= real_wf["포트 MDD"]).mean()]},
                  index=["포트 샤프", "포트 CAGR", "포트 MDD"])
wp.index.name = "2006–2026 포트폴리오"
log(md_table(wp))
log()
fig, ax = plt.subplots(figsize=(11, 4))
m = idx >= "2006-01-01"
for n, r in wf.items():
    en = {"워크포워드 기울기 전환": "WF slope-switch", "워크포워드 단일 이평": "WF single MA",
          "지난 다구치 최적(고정)": "Taguchi fixed (EMA100)", "매수 후 보유": "Buy & Hold"}[n]
    ax.plot(idx[m], np.cumprod(1 + np.nan_to_num(np.nanmean(r[m], axis=1))), label=en)
ax.set_yscale("log")
ax.grid(alpha=0.3)
ax.legend()
ax.set_title("Walk-forward 2006-2026, 14-ETF equal-weight portfolio")
fig.tight_layout()
fig.savefig(OUT / "walkforward.png", dpi=110)
plt.close(fig)

# ------------------------------------------------------------------ 정합성 점검
fx_exact = bt.evaluate(bt.returns(POS[f_b][br["fixed_len"]]), IS_PERIODS)["sharpe"].ravel()
log("## 부록. 고속 평가기 정합성")
log()
log(f"- 단일 이평 {LENS[br['fixed_len']]}일 로버스트 점수: 고속 {br['fixed_robust']:.4f} vs 정확 "
    f"{np.nanmean(fx_exact) - np.nanstd(fx_exact, ddof=1):.4f}")
sw_exact = bt.evaluate(bt.returns(sw_pos), IS_PERIODS)["sharpe"].ravel()
log(f"- 전환 규칙 로버스트 점수: 고속 {br['robust']:.4f} vs 정확(전환 비용 포함) "
    f"{np.nanmean(sw_exact) - np.nanstd(sw_exact, ddof=1):.4f}")
log()
log("![국면별 이평선 진단](regime_ma_heatmap.png)")
log()
log("![위약 검정](placebo.png)")
log()
log("![워크포워드](walkforward.png)")
(OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
tab.to_csv(OUT / "settings_search.csv")
pt.to_csv(OUT / "placebo_by_setting.csv")
wl.to_csv(OUT / "walkforward_choices.csv")
print("saved →", OUT)
