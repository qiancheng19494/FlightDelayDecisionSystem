"""复用第二阶段映射与标签定义，不将字符串 nan 当成真实实体。"""
import json
import pandas as pd
from core.field_mapper import FieldMapper
from core.time_parser import parse_dates

def normalize_key(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.upper()
    return text.mask(text.isin(["", "NAN", "NONE", "NULL", "<NA>"]))

def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = FieldMapper().map_fields(list(frame.columns))
    if not mapping.is_valid:
        raise ValueError("历史数据/查询缺少必要字段或字段冲突：" + ", ".join(mapping.missing_required_fields + mapping.conflicts))
    result = frame.rename(columns=mapping.mapped_fields).copy()
    if pd.api.types.is_datetime64_any_dtype(result["flight_date"]):
        result["flight_date"] = result["flight_date"].dt.normalize().dt.tz_localize(None)
    else:
        result["flight_date"] = parse_dates(result["flight_date"])
    for name in ["airline", "origin", "destination"]:
        result[name] = normalize_key(result[name])
    return result

def entity_keys(frame: pd.DataFrame, columns: tuple) -> pd.Series:
    """JSON 元组编码保留路线方向，并避免连接符导致的组合键冲突。"""
    if not columns:
        return pd.Series("[]", index=frame.index, dtype="string")
    part = frame[list(columns)]
    valid = part.notna().all(axis=1)
    keys = pd.Series(pd.NA, index=frame.index, dtype="string")
    keys.loc[valid] = [json.dumps(list(row), ensure_ascii=False, separators=(",", ":")) for row in part.loc[valid].itertuples(index=False, name=None)]
    return keys
