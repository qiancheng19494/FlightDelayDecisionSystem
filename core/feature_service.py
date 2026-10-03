"""GUI 与脚本共享的有界样本任务；完整计划表按块聚合，不导出全矩阵。"""
from pathlib import Path
from dataclasses import asdict
import json
from uuid import uuid4
import pandas as pd
from core.data_loader import FlightDataLoader
from core.historical_features import HistoricalFeatureEngine,json_safe
from core.history_cache import file_fingerprint
from core.feature_groups import FeatureCombinationRegistry,FEATURE_DEFINITIONS
from core.feature_pipeline import FeaturePipeline
from core.congestion_features import CongestionFeatureEngineer
from core.feature_validator import FeatureValidator

def build_feature_sample(context,dataset_id,history_id,combination,schedule_id=None,limit=100,progress=None,cancelled=None):
    if not 1<=limit<=5000:raise ValueError("特征样本上限为 5000 行")
    c=FeatureCombinationRegistry().get_combination(combination)
    def selected(identity):
        r=context.datasets.repository.get(identity)
        if context.datasets.source_status(r)!="READY":raise ValueError("源数据已变化或不可访问，请先重新分析")
        return r
    source=selected(dataset_id);fingerprints={source["file_path"]:file_fingerprint(Path(source["file_path"]))}
    loader=FlightDataLoader(source["file_path"],context.config.get("csv_chunksize"))
    parts=[];n=0
    for chunk in loader.iter_chunks(cancelled=cancelled):
        part=chunk.head(limit-n);parts.append(part);n+=len(part)
        if n>=limit:break
    if not parts:raise ValueError("所选数据集没有记录可预览")
    target=pd.concat(parts,ignore_index=True)
    if progress:progress(15,n)
    history=None
    if "H" in c.groups:
        if history_id is None:raise ValueError("请选择已构建的历史缓存")
        history=HistoricalFeatureEngine(context)
        if not history.load_cache(history_id):raise ValueError("历史缓存不存在或失效，请先在历史运行统计页构建")
    schedule=None;schedule_record=None
    if "C" in c.groups and schedule_id is not None:
        schedule_record=selected(schedule_id);path=Path(schedule_record["file_path"]);fingerprints[str(path)]=file_fingerprint(path)
        schedule=CongestionFeatureEngineer().fit_chunks(FlightDataLoader(path,context.config.get("csv_chunksize")).iter_chunks(cancelled=cancelled))
    if cancelled and cancelled():raise ValueError("任务已取消")
    if progress:progress(75,n)
    mode="TRAIN" if source["dataset_type"]=="HISTORICAL_LABELED" else "INFERENCE"
    result=FeaturePipeline(history).build_features(target,c,mode=mode,schedule_context=schedule,dataset_id=dataset_id)
    result.metadata["schedule_dataset_id"]=schedule_id if schedule_record else None
    result.metadata["schedule_context_rows"]=schedule.context_rows if schedule else 0
    result.metadata["source_fingerprints"]=fingerprints
    result.metadata["sample_limit"]=limit
    result.metadata["dataset_name"]=source["dataset_name"]
    result.metadata["is_demo"]=bool(source["is_demo"])
    for path,fingerprint in fingerprints.items():
        if file_fingerprint(Path(path))!=fingerprint:raise ValueError("特征构建期间源文件发生变化，请重试")
    if progress:progress(100,n)
    return result

def feature_profile(result):
    return pd.DataFrame([{"字段":n,"来源组":FEATURE_DEFINITIONS[n].group,"类型":str(result.X[n].dtype),"缺失率":f"{result.X[n].isna().mean():.1%}","样例":str(result.X[n].dropna().iloc[0]) if result.X[n].notna().any() else "暂无值","含义":FEATURE_DEFINITIONS[n].description} for n in result.X])

def audit_result(result):
    report=FeatureValidator().validate(result.X,result.metadata["combination_id"],lineage=result.metadata["lineage"],schema_version=result.metadata["feature_schema_version"],raise_on_error=False)
    report["checks"].append({"name":"X/y 行数与索引分离","passed":result.y is None or (len(result.y)==len(result.X) and result.y.index.equals(result.X.index) and result.y.name not in result.X)})
    if "H" in result.metadata["groups"]:
        from core.leakage_auditor import TemporalLeakageAuditor
        from core.history_schema import HistoricalBatchResult
        from core.feature_groups import HISTORY_FEATURES
        passed=False
        if result.history_metadata is not None and set(HISTORY_FEATURES).issubset(result.X):
            passed=TemporalLeakageAuditor().audit_history_features(HistoricalBatchResult(result.X[list(HISTORY_FEATURES)],result.history_metadata))["passed"]
        report["checks"].append({"name":"H 引擎实际 strict-date 审计","passed":passed})
    report["passed"]=all(c["passed"] for c in report["checks"])
    return report

def export_feature_sample(result,directory):
    if len(result.X)>5000:raise ValueError("样本导出最多 5000 行")
    report=audit_result(result)
    if not report["passed"]:raise ValueError("特征审计未通过，禁止导出")
    destination=Path(directory)/("features_"+uuid4().hex[:12]);destination.mkdir(parents=True,exist_ok=False)
    try:
        result.X.to_csv(destination/"X_features.csv",index=False,encoding="utf-8-sig")
        if result.y is not None:result.y.to_csv(destination/"y_labels.csv",index=False,encoding="utf-8-sig")
        payload={**result.metadata,"warnings":result.warnings,"feature_audit":report,"row_alignment":"X 与 y 按文件行位置一一对应；没有隐式丢行"}
        (destination/"metadata.json").write_text(json.dumps(json_safe(payload),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    except Exception:
        (destination/"INCOMPLETE.txt").write_text("导出失败；本目录不可作为完整特征样本使用。",encoding="utf-8")
        raise
    return destination
