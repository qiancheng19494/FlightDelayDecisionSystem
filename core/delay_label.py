"""只为具备有效延误数值的未取消航班生成监督标签。"""
import pandas as pd
from core.time_parser import numeric

def build_delay_label(departure_delay: pd.Series, cancelled: pd.Series | None = None) -> pd.Series:
    """缺取消字段时暂按未取消统计并由报告警告；存在但值未知则不标注。"""
    delay = numeric(departure_delay)
    eligible = delay.notna()
    if cancelled is not None:
        eligible &= numeric(cancelled).eq(0)
    labels = pd.Series(pd.NA, index=departure_delay.index, dtype="Int64")
    labels.loc[eligible] = delay.loc[eligible].ge(15).astype(int)
    return labels
