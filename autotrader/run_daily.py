"""하루 한 번 실행: 데이터 → 신호 → 주문 → 기록.

사용 예
  python -m autotrader.run_daily --config autotrader/config.json            # 설정대로 (기본 paper)
  python -m autotrader.run_daily --config autotrader/config.json --dry-run  # 주문 없이 신호만 확인

실행 시점: 미국장 마감 후(한국시간 새벽 6시 이후) 실행 → 그날 종가로 신호 판정 →
다음 거래일에 지정가 주문(한국투자증권 미국주식 예약주문 시간대 활용).
백테스트의 '신호 다음날 체결'과 같은 흐름이다.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from autotrader.broker import KISBroker, PaperBroker
from autotrader.data import load_prices
from autotrader.rebalance import make_orders
from autotrader.strategy import Params, signals


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="autotrader/config.json")
    ap.add_argument("--dry-run", action="store_true", help="주문을 내지 않고 신호와 예정 주문만 출력")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    tickers = list(cfg["universe"].keys())
    p = Params(**cfg["params"])
    if p.leverage > 1 and cfg["mode"] != "paper":
        raise SystemExit("레버리지 >1 은 가상계좌(paper)에서만 허용. 실계좌는 신용/증거금 조건을 직접 확인할 것.")

    price, bench = load_prices(tickers, cfg.get("bench", "SPY"), cfg.get("data_source", "yfinance"))
    sig = signals(price, bench, p)
    d = sig.attrs["date"]
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] 기준일 {d.date()}  SPY {sig.attrs['bench_close']:.2f} / "
          f"SMA{p.trend_len} {sig.attrs['bench_trend_line']:.2f}  장기추세 {'ON(상승)' if sig.attrs['trend_ok'] else 'OFF(하락→전량현금)'}")
    print(sig[["close", "ma", "buy_line", "sell_line", "hold", "changed", "target_weight"]].round(2).to_string())

    if cfg["mode"] == "paper":
        broker = PaperBroker(cfg.get("paper_state", "autotrader/state/paper.json"), cfg["budget_usd"], cfg.get("cost", 0.001))
    elif cfg["mode"] in ("kis_mock", "kis_live"):
        broker = KISBroker(cfg["universe"], mock=cfg["mode"] == "kis_mock")
    else:
        raise SystemExit(f"알 수 없는 mode: {cfg['mode']}")

    holdings = broker.positions()
    cash = broker.cash() if hasattr(broker, "cash") else None
    orders = make_orders(sig, holdings, cfg["budget_usd"], cfg.get("limit_buffer", 0.005), cash)
    print(f"\n보유: {holdings or '없음'}")
    print(f"예정 주문 {len(orders)}건:" if orders else "\n오늘은 주문 없음 (신호 변화 없음)")
    for o in orders:
        print(f"  {o.side.upper():4} {o.ticker:5} {o.qty:5}주  지정가 {o.price:.2f}")
    ordered = {o.ticker for o in orders}
    skipped = [t for t in sig.index[sig["hold"]] if holdings.get(t, 0) == 0 and t not in ordered]
    if skipped:
        need = max(sig.loc[skipped, "close"]) * len(sig) / p.leverage
        print(f"⚠ 보유 신호지만 1주도 살 수 없어 건너뜀: {skipped} "
              f"(종목당 배정액 < 1주 가격, 또는 현금 부족). 예산을 약 ${need:,.0f} 이상으로 늘리거나 종목 수를 줄이세요.")
    if a.dry_run or not orders:
        return
    if cfg["mode"] == "kis_live" and not cfg.get("i_understand_live_trading", False):
        raise SystemExit("실전 주문 차단: config 에 \"i_understand_live_trading\": true 를 직접 넣어야 실전 주문이 나갑니다.")

    log = Path(cfg.get("log_file", "autotrader/state/orders.csv"))
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "signal_date", "mode", "ticker", "side", "qty", "price", "ok", "msg"])
        for o in orders:
            r = broker.place(o)
            print(f"  → {r['msg']}")
            w.writerow([datetime.now().isoformat(timespec="seconds"), d.date(), cfg["mode"],
                        o.ticker, o.side, o.qty, o.price, r["ok"], r["msg"]])


if __name__ == "__main__":
    main()
