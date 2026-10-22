"""Tests for the stock_quote tool (Yahoo Finance chart endpoint)."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from openjarvis.security.capabilities import (
    DEFAULT_TOOL_CAPABILITIES,
    Capability,
)
from openjarvis.tools.stock_quote import StockQuoteTool


def _chart(meta=None, error=None):
    result = [{"meta": meta}] if meta is not None else None
    return {"chart": {"result": result, "error": error}}


_AAPL = {
    "symbol": "AAPL",
    "longName": "Apple Inc.",
    "currency": "USD",
    "regularMarketPrice": 190.5,
    "chartPreviousClose": 188.0,
    "regularMarketDayHigh": 191.0,
    "regularMarketDayLow": 187.25,
    "regularMarketTime": 1761148800,  # 2025-10-22 16:00 UTC
    "fullExchangeName": "NasdaqGS",
}


def _respond(routes):
    """Fake httpx.get keyed on the symbol in the URL path."""

    def fake_get(url, **kwargs):
        symbol = url.rsplit("/", 1)[-1]
        status, body = routes[symbol]
        if isinstance(status, Exception):
            raise status
        return httpx.Response(status, json=body, request=httpx.Request("GET", url))

    return fake_get


def test_quote_reports_price_change_range_and_delay_notice():
    with patch(
        "openjarvis.tools.stock_quote.httpx.get",
        side_effect=_respond({"AAPL": (200, _chart(_AAPL))}),
    ) as get:
        result = StockQuoteTool().execute(symbols="aapl")

    assert result.success
    assert "AAPL (Apple Inc.): 190.50 USD" in result.content
    assert "+2.50 (+1.33%) vs previous close 188.00" in result.content
    assert "Day range 187.25 - 191.00" in result.content
    assert "2025-10-22 16:00 UTC on NasdaqGS" in result.content
    assert "may be delayed" in result.content
    kwargs = get.call_args.kwargs
    assert kwargs["headers"]["User-Agent"]
    assert kwargs["follow_redirects"] is False


def test_partial_failures_are_reported_per_symbol():
    routes = {
        "AAPL": (200, _chart(_AAPL)),
        "NOPE": (404, {}),
        "SLOW": (httpx.ReadTimeout("t"), None),
        "BAD": (200, _chart(error={"code": "Not Found", "description": "No data"})),
    }
    with patch("openjarvis.tools.stock_quote.httpx.get", side_effect=_respond(routes)):
        result = StockQuoteTool().execute(symbols="AAPL, nope, SLOW, bad, AAPL")

    assert result.success
    lines = result.content.splitlines()
    assert lines[0].startswith("AAPL (Apple Inc.)")
    assert "NOPE: symbol not found." in lines
    assert "SLOW: the quote service timed out." in lines
    assert "BAD: No data." in lines
    assert sum(line.startswith("AAPL") for line in lines) == 1


def test_all_failures_mark_result_unsuccessful_without_source_line():
    with patch(
        "openjarvis.tools.stock_quote.httpx.get",
        side_effect=_respond({"ZZZ": (429, {})}),
    ):
        result = StockQuoteTool().execute(symbols="ZZZ")

    assert not result.success
    assert "rate limited" in result.content
    assert "Source:" not in result.content


@pytest.mark.parametrize("symbols", ["", " , ", "../etc/passwd", "A" * 30])
def test_invalid_input_never_reaches_the_network(symbols):
    with patch("openjarvis.tools.stock_quote.httpx.get") as get:
        result = StockQuoteTool().execute(symbols=symbols)
    assert not result.success
    get.assert_not_called()


def test_too_many_symbols_is_rejected():
    with patch("openjarvis.tools.stock_quote.httpx.get") as get:
        result = StockQuoteTool().execute(symbols=",".join(f"S{i}" for i in range(11)))
    assert not result.success
    get.assert_not_called()


def test_tool_is_loaded_with_network_capability_only():
    import sys

    import openjarvis.tools  # noqa: F401

    # The package import is what registers built-in tools.
    assert "openjarvis.tools.stock_quote" in sys.modules
    assert StockQuoteTool().spec.name == "stock_quote"
    assert DEFAULT_TOOL_CAPABILITIES["stock_quote"] == [Capability.NETWORK_FETCH]
