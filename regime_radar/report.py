"""
Reporting utilities for summarizing detected regimes
"""
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .model import Regime, SegmentationResult

@dataclass
class RegimeSummary:
    index: int
    start_date: date
    end_date: date
    length: int
    mean_daily: float
    vol_daily: float
    mean_annual: float
    vol_annual: float

@dataclass
class Report:
    ticker: str
    start_date: date
    end_date: date
    n_observations: int
    penalty: float
    total_cost: float
    regimes: List[RegimeSummary]

def build_report(ticker: str, df: pd.DataFrame, segmentation: SegmentationResult) -> Report:
    """
    Creates a report object from the df and segmentation result
    
    Params:
        ticker: Ticker symbol
        df: DataFrame with index as DatetimeIndex and "log_return" column
        segmentation: Output of model method
    
    Returns: 
        Report: Going to be turned into JSON
    """
    if df.empty:
        raise ValueError("Cannot build report from an empty DataFrame.")
    
    start_date = df.index[0].date()
    end_date = df.index[-1].date()
    n_obs = len(df)

    regimes_summaries: list[RegimeSummary] = []

    for idx, regime in enumerate(segmentation.regimes, start=1):
        seg_df = df.iloc[regime.start : regime.end + 1]
        seg_start = seg_df.index[0].date()
        seg_end = seg_df.index[-1].date()
        seg_len = len(seg_df)

        mean_daily = float(seg_df["log_return"].mean())
        vol_daily = float(seg_df["log_return"].std(ddof=1))

        mean_annual = float(mean_daily * 252)
        vol_annual = float(vol_daily * np.sqrt(252))

        regimes_summaries.append(
            RegimeSummary(
                index=idx,
                start_date=seg_start,
                end_date=seg_end,
                length=seg_len,
                mean_daily=mean_daily,
                vol_daily=vol_daily,
                mean_annual=mean_annual,
                vol_annual=vol_annual
            )
        )
    return Report(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        n_observations=n_obs,
        penalty=segmentation.penalty,
        total_cost=segmentation.total_cost,
        regimes=regimes_summaries
    )

def report_to_dict(report: Report) -> Dict[str, Any]:
    """
    Covert Report to JSON serializable dict
    """
    def _convert(obj: Any) -> Any:
        if isinstance(obj, date):
            return obj.isoformat()
        
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _convert(v) for k, v in asdict(obj).items()}
        
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        
        return obj
    return _convert(report)

def dump_report(report: Report, path: str | Path) -> None:
    """
    Serialize Report to a JSON file at a given path
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = report_to_dict(report)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)