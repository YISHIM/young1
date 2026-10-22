"""data/ 폴더의 가격·금리 데이터를 최신으로 갱신.

- ETF 수정주가: 공개 데이터셋 github.com/Johnbrick123/sp500-data (매일 갱신, Parquet ~235MB)
- 무위험금리: 공개 데이터셋 github.com/SteelCerberus/us-market-data
"""
import io
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
PRICES = "https://github.com/Johnbrick123/sp500-data/releases/download/data/prices.parquet"
RF = "https://raw.githubusercontent.com/SteelCerberus/us-market-data/main/data/us_market_data.csv"
UNIVERSE = ["SPY", "QQQ", "DIA", "IWM", "MDY",
            "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as r:
        return r.read()


def main() -> None:
    (ROOT / "data").mkdir(exist_ok=True)
    df = pd.read_parquet(io.BytesIO(fetch(PRICES)), columns=["date", "ticker", "adj_close"])
    p = df[df.ticker.isin(UNIVERSE)].pivot(index="date", columns="ticker", values="adj_close")[UNIVERSE]
    p = p[p.index >= "1998-12-22"]
    p.round(4).to_csv(ROOT / "data/etf_adj_close.csv")
    m = pd.read_csv(io.BytesIO(fetch(RF)), usecols=["Date", "Risk Free Rate"])
    m = m[m.Date >= "1998-01-01"].rename(columns={"Date": "date", "Risk Free Rate": "rf_annual_pct"})
    m.to_csv(ROOT / "data/risk_free.csv", index=False)
    print(f"prices {p.index[0].date()} ~ {p.index[-1].date()}, rf ~ {m.date.iloc[-1]}")


if __name__ == "__main__":
    main()
