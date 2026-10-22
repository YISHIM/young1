# young1 — 미국 주식 이동평균 추세매매 연구

이 저장소는 장기추세 필터와 이동평균 돌파/이탈 규칙을 조합한 전략을 실험합니다. 매개변수는 **다구치 실험계획법(직교배열)**으로 적은 횟수의 백테스트만 돌려 최적화하고, 그 결과를 전수조사와 표본외(미래) 구간으로 검증합니다.

결과 리포트: [`results/REPORT.md`](results/REPORT.md)

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
