"""Explicit history visibility; reuse the existing strict-day engine unchanged."""
from enum import Enum
import pandas as pd
from core.history_normalization import normalize_frame

class HistoryVisibilityMode(str, Enum):
    ROLLING_ASOF = "ROLLING_ASOF"
    FROZEN_CUTOFF = "FROZEN_CUTOFF"

class EvaluationProtocol(str, Enum):
    RESEARCH_FROZEN = "RESEARCH_FROZEN"
    ONLINE_ROLLING = "ONLINE_ROLLING"

def cutoff_date(value):
    if value is None:raise ValueError("FROZEN_CUTOFF requires history_cutoff_date")
    try:date=pd.Timestamp(value)
    except (ValueError,TypeError):raise ValueError("Invalid history_cutoff_date") from None
    if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise ValueError("history_cutoff_date must be a valid timezone-free natural date")
    return date

def resolve_visibility(mode, visibility=None, cutoff=None):
    policy=HistoryVisibilityMode(visibility or ("FROZEN_CUTOFF" if mode in ("VALIDATION","TEST") else "ROLLING_ASOF"))
    if policy == HistoryVisibilityMode.FROZEN_CUTOFF:
        if mode == "TRAIN":raise ValueError("TRAIN must use expanding ROLLING_ASOF history")
        return policy,cutoff_date(cutoff).strftime("%Y-%m-%d")
    if cutoff is not None:raise ValueError("ROLLING_ASOF does not accept a cutoff")
    return policy,None

def protocol_metadata(protocol, train_end, validation_end):
    protocol=EvaluationProtocol(protocol)
    frozen=protocol == EvaluationProtocol.RESEARCH_FROZEN
    visibility="FROZEN_CUTOFF" if frozen else "ROLLING_ASOF"
    train_end=cutoff_date(train_end).strftime("%Y-%m-%d")
    validation_end=cutoff_date(validation_end).strftime("%Y-%m-%d")
    return dict(evaluation_protocol=protocol.value,history_visibility_mode=visibility,
                history_cutoff_date=train_end if frozen else None,
                train_history_policy="ROLLING_ASOF",validation_history_policy=visibility,
                validation_history_cutoff=train_end if frozen else None,
                test_history_policy=visibility,test_history_cutoff=validation_end if frozen else None)

def visible_history(engine, frame, visibility, cutoff):
    """Cutoff is inclusive; query cutoff+1 through the original strict-date engine.

    Keep actual target dates in audit metadata, including all raw/fallback provenance.
    Only the H query receives the substituted date, never A/T/C or labels.
    """
    target=normalize_frame(frame)
    if visibility == HistoryVisibilityMode.ROLLING_ASOF:
        result=engine.transform_chunk(target,audit=True)
    else:
        boundary=cutoff_date(cutoff)
        if target.flight_date.isna().any() or target.flight_date.le(boundary).any():
            raise ValueError("history cutoff must be strictly before every target date")
        query=target.copy();query["flight_date"]=boundary+pd.Timedelta(days=1)
        result=engine.transform_chunk(query,audit=True)
        result.metadata["target_date"]=target.flight_date.to_numpy()
    result.metadata["history_visibility_mode"]=visibility.value
    result.metadata["history_cutoff_date"]=cutoff
    engine.auditor.audit_history_features(result,raise_on_error=True)
    return result

def history_description(metadata):
    """Presentation-only legacy label: never edit old bundle or registry metadata."""
    if "evaluation_protocol" not in metadata:
        return "legacy_rolling_validation / ONLINE_ROLLING（旧 DEMO；原 metadata 未修改）"
    return (f"评价协议：{metadata['evaluation_protocol']} · 历史可见模式：{metadata['history_visibility_mode']}"
            f" · 历史截止日期：{metadata.get('history_cutoff_date') or '无（逐日滚动）'}")
