"""BTS 时间、日期和有限数值的安全解析。"""
from dataclasses import dataclass
import math
import pandas as pd

@dataclass(frozen=True)
class ParsedTime:
    text: str | None
    day_offset: int = 0

def parse_hhmm(value) -> ParsedTime:
    """2400 转换为次日 00:00，保留 day_offset=1；无效值返回空。"""
    try:
        number = float(value)
        if not math.isfinite(number) or number != int(number):
            return ParsedTime(None)
        number = int(number)
        if number == 2400:
            return ParsedTime("00:00", 1)
        hour, minute = divmod(number, 100)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return ParsedTime(f"{hour:02d}:{minute:02d}")
    except (ValueError, TypeError):
        pass
    return ParsedTime(None)

def parse_dates(values: pd.Series) -> pd.Series:
    """明确支持年-月-日、年/月/日、美式月/日/年；不猜测日/月/年。"""
    text = values.astype("string").str.strip()
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for pattern in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"]:
        mask = result.isna()
        result.loc[mask] = pd.to_datetime(text.loc[mask], format=pattern, errors="coerce")
    return result

def numeric(values: pd.Series) -> pd.Series:
    """无穷大和非法内容统一为缺失数值，原文仍保留在 CSV。"""
    result = pd.to_numeric(values, errors="coerce")
    return result.where(result.notna() & result.abs().lt(float("inf")))
