"""训练/推理共享特征逻辑。仅生成特征，不训练、不填充、不编码标签。"""
from dataclasses import dataclass,field,asdict
from datetime import datetime,timezone
import time
import numpy as np
import pandas as pd
from core.feature_groups import *
from core.history_normalization import normalize_frame
from core.time_parser import numeric
from core.temporal_features import TemporalFeatureEngineer,planned_minutes
from core.congestion_features import CongestionFeatureEngineer
from core.feature_validator import FeatureValidator
from core.delay_label import build_delay_label
from core.history_visibility import resolve_visibility,visible_history
@dataclass
class FeatureBuildResult:
    X: pd.DataFrame
    y: pd.Series | None
    metadata: dict
    warnings: list=field(default_factory=list)
    history_metadata: pd.DataFrame | None=None
class FeaturePipeline:
    def __init__(self,historical_engine=None):self.history=historical_engine
    def build_features(self,dataframe,combination="COMBO_ATHC",*,mode="INFERENCE",schedule_context=None,dataset_id=None,history_visibility_mode=None,history_cutoff_date=None):
        if mode not in ("TRAIN","INFERENCE","VALIDATION","TEST"):raise ValueError("未知 FeaturePipeline mode")
        visibility,cutoff=resolve_visibility(mode,history_visibility_mode,history_cutoff_date)
        start=time.perf_counter();c=FeatureCombinationRegistry().validate_combination(combination)
        original_index=dataframe.index.copy();f=normalize_frame(dataframe).reset_index(drop=True)
        chunks=[];warnings=[];timings={};hm=None;ha=None
        for g in c.groups:
            tick=time.perf_counter()
            if g=="A":
                x=pd.DataFrame(index=f.index)
                for name in BASE_FEATURES_A:
                    value=f.get(name,pd.Series(np.nan,index=f.index))
                    x[name]=value.astype("string") if name in ("airline","origin","destination") else (planned_minutes(value) if name in ("scheduled_departure","scheduled_arrival") else numeric(value))
            elif g=="T":x=TemporalFeatureEngineer().transform(f)
            elif g=="H":
                if self.history is None or self.history.metadata is None:raise ValueError("组合需要 H：请先构建并选择有效历史缓存；Pipeline 不自动重建")
                h=visible_history(self.history,f,visibility,cutoff);x=h.features;hm=h.metadata
                ha=self.history.auditor.audit_history_features(h,raise_on_error=True);warnings.extend(h.warnings)
            else:
                engineer=schedule_context if isinstance(schedule_context,CongestionFeatureEngineer) else CongestionFeatureEngineer()
                if schedule_context is not None and not isinstance(schedule_context,CongestionFeatureEngineer):engineer.fit(schedule_context)
                x=engineer.transform(f)
                if x.congestion_context_available.eq(0).any():warnings.append("部分目标日期缺少已知完整计划上下文：C 为 NaN / unavailable，不以单条航班冒充完整计划表。")
                if x[list(CONGESTION_KEYS)].isna().any().any():warnings.append("C 含缺失值：可能缺计划日期、时刻、实体键或对应日期上下文。")
            chunks.append(x);timings[g+"_seconds"]=round(time.perf_counter()-tick,6)
        X=pd.concat(chunks,axis=1) if chunks else pd.DataFrame(index=f.index)
        X=X.reindex(columns=c.feature_names);X.index=original_index
        y=None
        if mode in ("TRAIN","VALIDATION"):
            if "departure_delay" not in f:raise ValueError("TRAIN 需要 departure_delay 标签来源列")
            y=build_delay_label(f.departure_delay,f.get("cancelled"));y.index=original_index;y.name="delay_label"
            if "cancelled" not in f:warnings.append("缺少取消字段，标签无法核实取消状态。")
        lineage={n:asdict(FEATURE_DEFINITIONS[n]) for n in c.feature_names}
        audit=FeatureValidator().validate(X,c,lineage=lineage)
        dates=f.flight_date.dropna();hmeta=self.history.metadata if "H" in c.groups else None
        metadata={"dataset_id":dataset_id,"combination_id":c.combination_id,"groups":list(c.groups),"feature_schema_version":FEATURE_SCHEMA_VERSION,"historical_schema_version":hmeta.get("historical_feature_schema_version") if hmeta else None,"historical_cache_id":self.history.folder.name if hmeta else None,"historical_dataset_id":hmeta.get("dataset_id") if hmeta else None,"generated_at":datetime.now(timezone.utc).isoformat(),"source_start_date":str(dates.min().date()) if len(dates) else None,"source_end_date":str(dates.max().date()) if len(dates) else None,"row_count":len(X),"feature_count":len(X.columns),"group_feature_counts":{g:len(GROUP_FIELDS[g]) if g in c.groups else 0 for g in GROUP_FIELDS},"forbidden_field_check":audit["passed"],"validation":audit,"history_audit":ha,"lineage":lineage,"feature_names":list(X),"mode":mode,"paper_combinations_confirmed":False,"schedule_semantics":"explicit context, flight_date cohort; final outcomes ignored; completeness is caller assertion","timings":timings}
        metadata.update(history_visibility_mode=visibility.value,history_cutoff_date=cutoff)
        metadata["timings"]["total_seconds"]=round(time.perf_counter()-start,6)
        if hm is not None:hm.index=original_index
        return FeatureBuildResult(X,y,metadata,list(dict.fromkeys(warnings)),hm)
    def transform_single(self,record,combination="COMBO_ATHC",**kwargs):
        return self.build_features(pd.DataFrame([record]),combination,**kwargs)
