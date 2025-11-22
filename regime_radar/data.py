"""
Data fetching and preprocessing
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

import numpy as np
import pandas as pd 
import yfinance as yf

PriceField = Literal["close", "adj_close"]

@dataclass
class PriceSeries:
    """
    Container for price and return series.

    index: DatetimeIndex (trading days)
    columns:
        "price": selected price field (float)
        "log_return": log(P_t) - log(P_{t-1})
    """
    ticker: str
    data: pd.DataFrame # refer to columns above

def _normalize_dates(start: Optional[str], end: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Helper to validate and normalize date strings
    """
    def _check(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        try:
            datetime.strptime(s, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid date format {s!r}, expected YYYY-MM--DD") from exc
        return s
    return _check(start), _check(end)

def fetch_price_series(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    field: PriceField = "adj_close",
) -> PriceSeries:
    """
    Fetch daily price data for ticker and return PriceSeries with prices and log returns.

    Params:
        ticker: Ticker symbol, e.g. "AAPL", "SPY"
        start, end: Date boundaries (YYYY-MM-DD), if not inputted then use full history
        field: Which price field to use, adjusted close or raw close
    
    Return:
        PriceSeries: df indexed by date and price and log return columns
    """
    start, end = _normalize_dates(start, end)

    # yfin returns df w/ columns like "Open", "High", ...
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker!r} in the given range")
    
    if field == "adj_close":
        col = "Adj Close"
    elif field == "close":
        col = "Close"
    else:
        raise ValueError(f"Unsupported field {field!r}")
    
    if col not in df.columns:
        raise ValueError(f"Expected column {col!r} not found in downloaded data")
    
    prices = df[col].dropna()

    if len(prices) < 2:
        raise ValueError(f"No valid price data for ticker {ticker!r} after removing NaNs. Try a longer date range or different ticker")
    
    # log_ret: log returns ( log[P_t] - log[P_{t-1}] )
    log_prices = np.log(prices)
    log_ret = log_prices.diff().dropna()


    result = pd.DataFrame(index=prices.index.copy())
    result["price"] = prices
    result["log_return"] = log_ret
    result = result.dropna()

    if result.empty:
        raise ValueError(f"Insufficient data to compute log returns for ticker {ticker!r}")
    
    if not isinstance(result.index, pd.DatetimeIndex):
        result.inidex = pd.to_datetime(result.index)

    return PriceSeries(ticker=ticker.upper(), data=result)