"""Stock quote tool -- latest price and daily change from Yahoo Finance.

Uses Yahoo Finance's public chart endpoint, which needs no API key. Quotes can
be delayed (typically up to 15 minutes, depending on the exchange), so results
say so explicitly. Symbols follow Yahoo's conventions: ``AAPL``, ``BHP.AX``
(ASX), ``^GSPC`` (S&P 500), ``BTC-USD``, ``AUDUSD=X``.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import httpx

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

_TOOL_NAME = "stock_quote"
_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
# Yahoo rejects requests without a browser-like User-Agent (HTTP 429).
_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9.\-^=]{1,20}$")
_MAX_SYMBOLS = 10


class StockQuoteError(RuntimeError):
    """A user-presentable failure fetching one symbol."""


def _fetch_meta(symbol: str) -> dict[str, Any]:
    try:
        resp = httpx.get(
            _CHART_URL.format(symbol=symbol),
            params={"range": "1d", "interval": "5m"},
            headers=_HEADERS,
            timeout=15.0,
            follow_redirects=False,
        )
    except httpx.TimeoutException:
        raise StockQuoteError("the quote service timed out") from None
    except httpx.HTTPError as exc:
        raise StockQuoteError(f"could not reach the quote service ({exc})") from None

    if resp.status_code == 404:
        raise StockQuoteError("symbol not found")
    if resp.status_code == 429:
        raise StockQuoteError("rate limited by the quote service; try again shortly")
    if resp.status_code >= 300:
        raise StockQuoteError(f"quote service returned HTTP {resp.status_code}")

    try:
        chart = resp.json()["chart"]
    except (ValueError, KeyError, TypeError):
        raise StockQuoteError("unexpected response from the quote service") from None
    if chart.get("error"):
        description = (chart["error"] or {}).get("description") or "symbol not found"
        raise StockQuoteError(str(description))
    results = chart.get("result") or []
    meta = results[0].get("meta") if results else None
    if not meta or meta.get("regularMarketPrice") is None:
        raise StockQuoteError("no price available for this symbol")
    return meta


def _format_quote(symbol: str, meta: dict[str, Any]) -> str:
    price = float(meta["regularMarketPrice"])
    currency = meta.get("currency") or ""
    name = meta.get("longName") or meta.get("shortName")
    label = f"{meta.get('symbol') or symbol}" + (f" ({name})" if name else "")
    line = f"{label}: {price:,.2f} {currency}".rstrip()

    prev = meta.get("chartPreviousClose") or meta.get("previousClose")
    if prev:
        change = price - float(prev)
        pct = change / float(prev) * 100
        line += f", {change:+,.2f} ({pct:+.2f}%) vs previous close {float(prev):,.2f}"

    high, low = meta.get("regularMarketDayHigh"), meta.get("regularMarketDayLow")
    if high is not None and low is not None:
        line += f". Day range {float(low):,.2f} - {float(high):,.2f}"

    stamp = meta.get("regularMarketTime")
    exchange = meta.get("fullExchangeName") or meta.get("exchangeName")
    when = ""
    if isinstance(stamp, (int, float)):
        when = datetime.fromtimestamp(stamp, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    if when or exchange:
        line += ". As of " + " on ".join(p for p in (when, exchange) if p)
    return line + "."


@ToolRegistry.register("stock_quote")
class StockQuoteTool(BaseTool):
    """Latest price and daily change for stocks, indices, FX and crypto."""

    tool_id = _TOOL_NAME

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=_TOOL_NAME,
            description=(
                "Get the latest market price and today's change for one or more"
                " ticker symbols (stocks, ETFs, indices, FX, crypto). Use Yahoo"
                " Finance symbols: AAPL, TSLA, BHP.AX for ASX shares, ^GSPC for"
                " the S&P 500, ^AXJO for the ASX 200, BTC-USD, AUDUSD=X."
                " Quotes may be delayed by up to about 15 minutes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "string",
                        "description": (
                            "Comma-separated ticker symbols, e.g. 'AAPL, NVDA, ^GSPC'"
                        ),
                    },
                },
                "required": ["symbols"],
            },
            category="finance",
        )

    def execute(self, **params: Any) -> ToolResult:
        raw = str(params.get("symbols") or params.get("symbol") or "")
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        if not symbols:
            return ToolResult(
                tool_name=_TOOL_NAME,
                content="No ticker symbol provided.",
                success=False,
            )
        if len(symbols) > _MAX_SYMBOLS:
            return ToolResult(
                tool_name=_TOOL_NAME,
                content=f"Too many symbols; ask for at most {_MAX_SYMBOLS} at once.",
                success=False,
            )

        lines: list[str] = []
        ok = 0
        for symbol in dict.fromkeys(symbols):
            if not _SYMBOL_PATTERN.match(symbol):
                lines.append(f"{symbol}: not a valid ticker symbol.")
                continue
            try:
                lines.append(_format_quote(symbol, _fetch_meta(symbol)))
                ok += 1
            except StockQuoteError as exc:
                lines.append(f"{symbol}: {exc}.")
        if ok:
            lines.append("Source: Yahoo Finance; prices may be delayed.")
        return ToolResult(
            tool_name=_TOOL_NAME,
            content="\n".join(lines),
            success=ok > 0,
        )


__all__ = ["StockQuoteTool"]
