"""Bounded offline BTS quality and compatibility; never equate schema with real origin."""
from pathlib import Path
import numpy as np
import pandas as pd
from core.data_loader import FlightDataLoader
from core.field_mapper import FieldMapper
from core.history_normalization import normalize_frame
from core.time_parser import parse_hhmm,numeric
from core.delay_label import build_delay_label
from core.feature_policy import PRE_FLIGHT_ALLOWED_FIELDS,POST_EVENT_FORBIDDEN_FIELDS

BTS_FIELDS=("flight_date","airline","origin","destination","scheduled_departure","scheduled_arrival","scheduled_elapsed_time","distance","departure_delay","cancelled")

def quality_frame(raw):
    mapping=FieldMapper().map_fields(list(raw));m=mapping.to_dict()
    result=dict(mapping=m,compatible=mapping.is_valid,sampled_rows=len(raw),unknown_fields=mapping.unknown_fields,
                prediction_allowed_fields=sorted(set(mapping.mapped_fields.values()) & PRE_FLIGHT_ALLOWED_FIELDS),
                posterior_fields=sorted(set(mapping.mapped_fields.values()) & POST_EVENT_FORBIDDEN_FIELDS),label_fields=[c for c in ("departure_delay","cancelled") if c in mapping.mapped_fields.values()],
                missing_bts_fields=[c for c in BTS_FIELDS if c not in mapping.mapped_fields.values()],dtypes={c:str(raw[c].dtype) for c in raw},scheduled_times={})
    if not mapping.is_valid:return result
    f=normalize_frame(raw);dates=f.flight_date.dropna()
    result.update(invalid_dates=int(f.flight_date.isna().sum()),date_range=[str(dates.min().date()),str(dates.max().date())] if len(dates) else None,
                  missing={c:int(raw[c].isna().sum()+raw[c].astype("string").str.strip().eq("").fillna(False).sum()) for c in raw})
    for field in ("scheduled_departure","scheduled_arrival"):
        if field in f:
            missing=f[field].isna()|f[field].astype("string").str.strip().eq("").fillna(False)
            parsed=[parse_hhmm(v) for v in f[field]];bad=pd.Series([p.text is None for p in parsed],index=f.index)
            result["scheduled_times"][field]=dict(invalid_or_missing=int(bad.sum()),invalid_nonempty=int((bad&~missing).sum()),missing=int(missing.sum()),next_day_2400=sum(p.day_offset for p in parsed))
    if "cancelled" in f:
        v=numeric(f.cancelled);missing=f.cancelled.isna()|f.cancelled.astype("string").str.strip().eq("").fillna(False)
        result["cancelled_check"]=dict(present=True,zero=int(v.eq(0).sum()),one=int(v.eq(1).sum()),missing=int(missing.sum()),other=int((~v.isin([0,1])&~missing).sum()),invalid_or_missing=int((~v.isin([0,1])).sum()))
    else:result["cancelled_check"]=dict(present=False,invalid_or_missing=None)
    if "departure_delay" in f:
        d=numeric(f.departure_delay);y=build_delay_label(f.departure_delay,f.get("cancelled"))
        result["labels"]=dict(valid=int(y.notna().sum()),delay=int(y.eq(1).sum()),normal=int(y.eq(0).sum()),invalid_delay=int(d.isna().sum()),delay_rate=float(y.mean()) if y.notna().any() else None)
        result["departure_delay"]=dict(dtype=str(f.departure_delay.dtype),missing_or_invalid=int(d.isna().sum()),negative=int(d.lt(0).sum()),zero=int(d.eq(0).sum()),equal_14=int(d.eq(14).sum()),equal_15=int(d.eq(15).sum()),above_15=int(d.gt(15).sum()),extreme_abs_gt_1440=int(d.abs().gt(1440).sum()),minimum=float(d.min()) if d.notna().any() else None,maximum=float(d.max()) if d.notna().any() else None)
        if "DepDel15" in raw:
            published=numeric(raw.DepDel15);valid=y.notna()&published.notna()
            result["published_label_check"]=dict(compared=int(valid.sum()),mismatches=int(y[valid].ne(published[valid]).sum()),scope="DEP_DELAY >=15 versus BTS DepDel15; no paper label changes")
    result["history_compatibility"]="READY_FOR_BOUNDED_SMOKE" if not result["invalid_dates"] and result.get("labels",{}).get("valid",0)>0 else "NOT_READY"
    result["feature_pipeline_compatibility"]="SCHEMA_READY_NOT_EXECUTED" if not result["missing_bts_fields"] else "MISSING_BTS_FIELDS"
    return result

def verify(path,limit=100):
    if not 1<=limit<=1000:raise ValueError("小样本行数须为 1–1000")
    loader=FlightDataLoader(path);raw=loader.read_preview(page_size=limit)
    return {**loader.file_info(),**quality_frame(raw),"original_columns":loader.original_columns,
            "bts_empty_trailing_column":loader.bts_trailing_empty,
            "scope":"仅前 N 行；字段兼容不证明真实来源，不代表完整数据或已运行 FeaturePipeline"}
