# young1 — 미국 주식 이동평균 추세매매 연구

이 저장소는 장기추세 필터와 이동평균 돌파/이탈 규칙을 조합한 전략을 실험합니다. 매개변수는 **다구치 실험계획법(직교배열)**으로 적은 횟수의 백테스트만 돌려 최적화하고, 그 결과를 전수조사와 표본외(미래) 구간으로 검증합니다.

## 연구 결과 요약

| 리포트 | 내용 | 결론 |
|---|---|---|
| [`results/REPORT.md`](results/REPORT.md) | 다구치 실험계획으로 이평선·돌파%·이탈%·장기추세 최적화 | EMA100, +1.5% 매수 / −4% 매도, SPY>SMA225 필터 (75회 실험으로 전수조사 상위 1.3%) |
| [`results/slope/REPORT.md`](results/slope/REPORT.md) | 장기추세 기울기에 따라 이평선을 바꾸는 규칙 | 기각 (위약 검정 p = 0.80) |
| [`results/ensemble/REPORT.md`](results/ensemble/REPORT.md) | 다중 이평 앙상블, 기울기별 비중 조절, 대형주 50종목 재검증 | 모두 기각. 성과의 원천은 장기추세 필터 |
| [`results/leverage/REPORT.md`](results/leverage/REPORT.md) | 낙폭이 줄어든 만큼 비중 확대 | 위험을 같게 맞추면 수익은 '그냥 보유'와 같고, 최대낙폭은 −53%에서 −34%로 얕아짐 |

## 자동매매 프로그램 (`autotrader/`)

하루 한 번, 미국장 마감 후(한국시간 오전 6시 이후) 실행합니다. 백테스트와 같은 신호 함수를 쓰고, 2019–2022 하루 단위 재생 시험에서 보유 종목이 매일 일치했습니다(`tests/`).

```bash
cp autotrader/config.example.json autotrader/config.json   # 예산·종목·모드 수정
python -m autotrader.run_daily --dry-run   # 신호와 예정 주문만 확인
python -m autotrader.run_daily             # 주문 실행 (기본: 내 컴퓨터 안의 가상계좌)
```

| mode | 동작 |
|---|---|
| `paper` (기본) | 가상계좌. `autotrader/state/` 에 잔고·주문 기록 |
| `kis_mock` | 한국투자증권 **모의투자** 서버. 환경변수 `KIS_APP_KEY`, `KIS_APP_SECRET`, `KIS_ACCOUNT` 필요 |
| `kis_live` | 실전. `"i_understand_live_trading": true` 를 직접 넣어야만 주문이 나감 |

- 한국투자증권 API 연동부는 공개 문서 기준으로 작성했고 실제 계정으로는 시험하지 않았습니다. 모의투자에서 먼저 확인하세요.
- 앱키와 계좌번호는 환경변수로만 넣고, 파일이나 저장소에 저장하지 마세요.
- 매일 자동 실행: Windows는 작업 스케줄러, Mac/Linux는 cron (예: `30 6 * * 2-6 cd ~/young1 && python -m autotrader.run_daily`).

## 매매 규칙

| 구성요소 | 내용 |
|---|---|
| 장기추세 기준종목 | SPY. 14종 투자대상군 전체 움직임과의 설명력 R²가 0.965로 가장 높음 |
| 장기추세 필터 | SPY 종가가 SPY의 N일 단순이평선 위에 있을 때만 매수 허용. 아래로 내려가면 전량 매도 |
| 매수 | 종목 종가가 자기 이평선보다 `buy%` 이상 위로 돌파 |
| 매도 | 종목 종가가 자기 이평선보다 `sell%` 이상 아래로 이탈 |
| 체결/비용 | 신호 다음 거래일 종가에 체결, 편도 0.1% 비용, 현금 보유 시 단기국채 이자 수취 |

투자대상군은 SPY, QQQ, DIA, IWM, MDY와 섹터 ETF 9종(XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU, XLB)입니다.

## 실행

```bash
pip install -r requirements.txt
python update_data.py      # (선택) 데이터 최신화
python run_experiment.py   # 실험 전체 실행 → results/
```

## 파일

- `trend_lab/backtest.py` — 벡터화 백테스트 엔진
- `trend_lab/taguchi.py` — L25 직교배열, 주효과, ANOVA, 최적조건 예측
- `run_experiment.py` — 기준종목 선정 → 1단계 실험 → 2단계 실험 → 확인실험 → 전수조사 검증 → 표본외 검증
- `data/` — ETF 수정주가(1998-12~), 무위험금리

> 과거 성과는 미래 수익을 보장하지 않습니다. 연구 및 학습용 코드입니다.
