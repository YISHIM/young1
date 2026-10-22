"""가격 데이터 불러오기. 실전에서는 yfinance(인터넷 필요), 오프라인 시험에서는 저장소의 CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_prices(tickers: list[str], bench: str, source: str = "yfinance", years: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    """수정주가(배당·분할 반영) 종가. 이평선 225일 계산을 위해 최소 2년 이상 필요."""
    allt = list(dict.fromkeys(tickers + [bench]))
    if source == "csv":
        df = pd.read_csv(ROOT / "data/etf_adj_close.csv", index_col=0, parse_dates=True)
        missing = [t for t in allt if t not in df.columns]
        if missing:
            raise ValueError(f"CSV에 없는 종목: {missing}")
        df = df[allt]
    elif source == "yfinance":
        import yfinance as yf  # pip install yfinance

        raw = yf.download(allt, period=f"{years}y", auto_adjust=True, progress=False)["Close"]
        df = raw[allt] if isinstance(raw, pd.DataFrame) else raw.to_frame(allt[0])
    else:
        raise ValueError(source)
    df = df.dropna(how="all")
    return df[tickers], df[bench]
