"""브로커 연결.

- PaperBroker : 내 컴퓨터 안의 가상 계좌 (JSON 파일에 저장). 체결은 주문 가격 그대로 + 비용.
- KISBroker   : 한국투자증권 Open API 해외주식. 기본은 모의투자 서버.
                API 사양은 KIS Developers 문서(apiportal.koreainvestment.com) 기준으로 작성했으며
                이 저장소에서는 실제 계정으로 시험하지 않았다. 반드시 모의투자 계좌로 먼저 확인할 것.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Order:
    ticker: str
    side: str        # "buy" | "sell"
    qty: int
    price: float     # 지정가


class PaperBroker:
    def __init__(self, state_file: str | Path, initial_cash: float, cost: float = 0.001):
        self.path = Path(state_file)
        self.cost = cost
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"cash": initial_cash, "positions": {}}

    def positions(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.state["positions"].items() if v}

    def cash(self) -> float:
        return float(self.state["cash"])

    def place(self, o: Order) -> dict:
        amt = o.qty * o.price
        fee = amt * self.cost
        pos = self.state["positions"]
        if o.side == "buy":
            if amt + fee > self.state["cash"] + 1e-9:
                return {"ok": False, "msg": "현금 부족"}
            self.state["cash"] -= amt + fee
            pos[o.ticker] = pos.get(o.ticker, 0) + o.qty
        else:
            if pos.get(o.ticker, 0) < o.qty:
                return {"ok": False, "msg": "보유 수량 부족"}
            self.state["cash"] += amt - fee
            pos[o.ticker] -= o.qty
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2))
        return {"ok": True, "msg": f"체결(가상) {o.side} {o.ticker} {o.qty}주 @ {o.price:.2f}"}


class KISBroker:
    """한국투자증권 해외주식(미국). 계좌·키는 환경변수로만 받는다(코드·설정 파일에 저장 금지).

    KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT (예: 12345678-01)
    """

    URL = {True: "https://openapivts.koreainvestment.com:29443",   # 모의투자
           False: "https://openapi.koreainvestment.com:9443"}       # 실전
    TR = {True: {"buy": "VTTT1002U", "sell": "VTTT1001U", "balance": "VTTS3012R"},
          False: {"buy": "TTTT1002U", "sell": "TTTT1006U", "balance": "TTTS3012R"}}
    # 시세 조회용 거래소 코드 / 주문용 거래소 코드
    QUOTE_EXCD = {"NASD": "NAS", "NYSE": "NYS", "AMEX": "AMS"}

    def __init__(self, exchanges: dict[str, str], mock: bool = True, token_file: str | Path = ".kis_token.json"):
        import requests  # pip install requests

        self.rq = requests
        self.mock = mock
        self.base = self.URL[mock]
        self.key = os.environ["KIS_APP_KEY"]
        self.secret = os.environ["KIS_APP_SECRET"]
        self.cano, self.prdt = os.environ["KIS_ACCOUNT"].split("-")
        self.exchanges = exchanges  # 종목 → 주문 거래소 코드 (NASD/NYSE/AMEX)
        self.token_file = Path(token_file)
        self.token = self._token()

    def _token(self) -> str:
        # 토큰은 하루 유효, 발급 횟수 제한이 있어 파일에 저장해 재사용
        if self.token_file.exists():
            t = json.loads(self.token_file.read_text())
            if t.get("mock") == self.mock and t["expires"] > time.time() + 600:
                return t["token"]
        r = self.rq.post(f"{self.base}/oauth2/tokenP", timeout=10,
                         json={"grant_type": "client_credentials", "appkey": self.key, "appsecret": self.secret})
        r.raise_for_status()
        d = r.json()
        self.token_file.write_text(json.dumps({"token": d["access_token"], "mock": self.mock,
                                               "expires": time.time() + int(d.get("expires_in", 86400))}))
        os.chmod(self.token_file, 0o600)
        return d["access_token"]

    def _headers(self, tr_id: str) -> dict:
        return {"content-type": "application/json; charset=utf-8", "authorization": f"Bearer {self.token}",
                "appkey": self.key, "appsecret": self.secret, "tr_id": tr_id, "custtype": "P"}

    def price(self, ticker: str) -> float:
        excd = self.QUOTE_EXCD[self.exchanges[ticker]]
        r = self.rq.get(f"{self.base}/uapi/overseas-price/v1/quotations/price", timeout=10,
                        headers=self._headers("HHDFS00000300"), params={"AUTH": "", "EXCD": excd, "SYMB": ticker})
        r.raise_for_status()
        return float(r.json()["output"]["last"])

    def positions(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ex in sorted(set(self.exchanges.values())):
            r = self.rq.get(f"{self.base}/uapi/overseas-stock/v1/trading/inquire-balance", timeout=10,
                            headers=self._headers(self.TR[self.mock]["balance"]),
                            params={"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "OVRS_EXCG_CD": ex,
                                    "TR_CRCY_CD": "USD", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""})
            r.raise_for_status()
            for row in r.json().get("output1", []):
                q = int(float(row.get("ovrs_cblc_qty", 0)))
                if q:
                    out[row["ovrs_pdno"]] = q
            time.sleep(0.5)  # 모의투자 초당 호출 제한
        return out

    def place(self, o: Order) -> dict:
        body = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "OVRS_EXCG_CD": self.exchanges[o.ticker],
                "PDNO": o.ticker, "ORD_QTY": str(o.qty), "OVRS_ORD_UNPR": f"{o.price:.2f}",
                "ORD_SVR_DVSN_CD": "0", "ORD_DVSN": "00"}  # 00 = 지정가
        r = self.rq.post(f"{self.base}/uapi/overseas-stock/v1/trading/order", timeout=10,
                         headers=self._headers(self.TR[self.mock][o.side]), json=body)
        d = r.json()
        time.sleep(0.5)
        return {"ok": d.get("rt_cd") == "0", "msg": d.get("msg1", r.text)}
