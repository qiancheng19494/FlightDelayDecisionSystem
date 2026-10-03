"""统一训练协调：输入→FeaturePipeline→审计→时间分区→train-only fit。"""
import json,time
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
from core.history_normalization import normalize_frame
from core.feature_pipeline import FeaturePipeline,FeatureBuildResult
from core.history_visibility import EvaluationProtocol,protocol_metadata
from core.feature_service import audit_result
from core.feature_groups import FeatureCombinationRegistry
from core.historical_features import HistoricalFeatureEngine,json_safe
from core.congestion_features import CongestionFeatureEngineer
from core.data_loader import FlightDataLoader
from core.history_cache import file_fingerprint
from core.ml.time_split import TemporalDataSplitter,LOCK_DATE
from core.ml.trainers import RandomForestTrainer,XGBoostTrainer
from core.ml.model_bundle import ModelBundle
from core.ml.model_manager import ModelManager,library_versions
from core.ml.model_registry import atomic_json
from core.ml.metrics import MetricsService

def timestamp():return datetime.now(timezone.utc).isoformat()
def load_model_config(root):
    path=Path(root)/"config/model_config.json"
    if not path.exists():path=Path(__file__).resolve().parents[2]/"config/model_config.json"
    config=json.loads(path.read_text(encoding="utf-8"))
    if config.get("default_threshold")!=.5:raise ValueError("Stage 5 threshold 必须固定 0.5")
    return config
class TrainingService:
    def __init__(self,root):self.root=Path(root);self.manager=ModelManager(root)
    def train_frame(self,frame,model_type,combination,*,history=None,schedule_context=None,is_demo=True,formal_requested=False,dataset_id=None,fingerprints=None,config=None,date_ranges=None,stage=None,evaluation_protocol=EvaluationProtocol.RESEARCH_FROZEN,training_purpose=None):
        config=config or load_model_config(self.root);stage=stage or (lambda text:None)
        if config["default_threshold"]!=.5:raise ValueError("Stage 5 threshold 固定 0.5")
        if model_type not in ("rf","xgboost"):raise ValueError("不支持的模型类型")
        if not is_demo and (not formal_requested or dataset_id is None or not fingerprints):raise ValueError("正式训练必须由用户明确启动，且提供真实数据集 ID / 来源指纹")
        evaluation_protocol=EvaluationProtocol(evaluation_protocol)
        if training_purpose not in (None,"REAL_SMOKE"):raise ValueError("Unknown training purpose")
        if training_purpose=="REAL_SMOKE" and is_demo:raise ValueError("REAL_SMOKE cannot use DEMO identity")
        combo=FeatureCombinationRegistry().validate_combination(combination)
        if not combo.feature_names:raise ValueError("当前组合不包含任何模型输入特征。")
        if len(frame)>config["max_training_rows"]:raise ValueError("本阶段内存训练最多 100000 行；不静默抽样")
        experiment_id="experiment_"+uuid4().hex;experiment_path=self.root/"output/experiments"/(experiment_id+".json")
        experiment={"experiment_id":experiment_id,"dataset_id":dataset_id,"model_type":model_type,"combination_id":combo.combination_id,"parameters":config,"random_state":config["random_state"],"started_at":timestamp(),"status":"STARTED","is_demo":is_demo,"evaluation_protocol":evaluation_protocol.value}
        atomic_json(experiment_path,experiment);total=time.perf_counter();saved_identity=None
        try:
            stage("划分时间并锁定最终评价集")
            dates=normalize_frame(frame).flight_date
            split=TemporalDataSplitter().split(dates,is_demo=is_demo,real_smoke=training_purpose=="REAL_SMOKE",**(date_ranges or {}))
            stage("生成特征并运行 Feature audit")
            feature_start=time.perf_counter()
            policies=protocol_metadata(evaluation_protocol,split.metadata["training_end_date"],split.metadata["validation_end_date"])
            frame=frame.reset_index(drop=True)
            pipeline=FeaturePipeline(history)
            train_result=pipeline.build_features(frame.iloc[split.train_positions],combo,mode="TRAIN",schedule_context=schedule_context,dataset_id=dataset_id,history_visibility_mode="ROLLING_ASOF")
            validation_result=pipeline.build_features(frame.iloc[split.validation_positions],combo,mode="VALIDATION",schedule_context=schedule_context,dataset_id=dataset_id,history_visibility_mode=policies["validation_history_policy"],history_cutoff_date=policies["validation_history_cutoff"])
            if not audit_result(train_result)["passed"] or not audit_result(validation_result)["passed"]:raise ValueError("Partition Feature audit failed")
            # Preserve original positional addressing, without generating excluded rows.
            positions=np.concatenate([split.train_positions,split.validation_positions])
            joined=pd.concat([train_result.X,validation_result.X]).sort_index()
            labels=pd.concat([train_result.y,validation_result.y]).sort_index()
            hm=pd.concat([train_result.history_metadata,validation_result.history_metadata]).sort_index() if "H" in combo.groups else None
            result=FeatureBuildResult(joined,labels,{**train_result.metadata,**policies},train_result.warnings+validation_result.warnings,hm)
            mapping={original:i for i,original in enumerate(sorted(positions))}
            train_positions=np.array([mapping[i] for i in split.train_positions]);validation_positions=np.array([mapping[i] for i in split.validation_positions])
            feature_time=time.perf_counter()-feature_start
            if not audit_result(result)["passed"]:raise ValueError("Feature audit 未通过，禁止训练")
            eligible=result.y.notna().to_numpy()
            train=train_positions[eligible[train_positions]];valid=validation_positions[eligible[validation_positions]]
            if not len(train) or not len(valid):raise ValueError("排除无效监督标签后训练/验证为空")
            trainer=(RandomForestTrainer if model_type=="rf" else XGBoostTrainer)(config).fit(result,train,stage)
            stage("计算验证集指标（固定阈值 0.50）")
            start=time.perf_counter();probability=trainer.predict_proba(result.X.iloc[valid]);predict_time=time.perf_counter()-start
            metrics=MetricsService.evaluate(result.y.iloc[valid].to_numpy(dtype=int),probability)
            metrics["label"]="DEMO VALIDATION METRICS" if is_demo else ("REAL_SMOKE VALIDATION METRICS" if training_purpose=="REAL_SMOKE" else "VALIDATION METRICS")
            for path,fingerprint in (fingerprints or {}).items():
                if file_fingerprint(Path(path))!=fingerprint:raise ValueError("训练期间源文件变化，拒绝保存")
            if history is not None:history._assert_fresh()
            identity=model_type+"_"+combo.combination_id.removeprefix("COMBO_").lower()+"_"+uuid4().hex
            hmeta={"dataset_id":history.dataset_id,"cache_id":history.folder.name,"configuration":history.config,"source_fingerprint":file_fingerprint(history.source_path)} if "H" in combo.groups else None
            metadata={"model_id":identity,"model_name":("DEMO MODEL " if is_demo else ("REAL_SMOKE " if training_purpose=="REAL_SMOKE" else "TRAINED MODEL "))+model_type.upper()+" "+combo.name,"model_type":model_type,"model_version":2,"is_demo":bool(is_demo),"status":"DEMO" if is_demo else ("REAL_SMOKE" if training_purpose=="REAL_SMOKE" else "TRAINED"),"training_purpose":training_purpose,"model_identity_notice":"真实 BTS 小样本工程验证模型，不代表论文最终结果。" if training_purpose=="REAL_SMOKE" else None,"created_at":timestamp(),"combination_id":combo.combination_id,"feature_schema_version":result.metadata["feature_schema_version"],"historical_schema_version":result.metadata["historical_schema_version"] if "H" in combo.groups else "not_required","feature_names":list(result.X),"feature_groups":list(combo.groups),**split.metadata,**policies,"training_row_count":len(train),"validation_row_count":len(valid),"excluded_invalid_label_rows":int((~eligible).sum()),"positive_rate_train":float(result.y.iloc[train].mean()),"positive_rate_validation":float(result.y.iloc[valid].mean()),"positive_count_train":trainer.positive_count,"negative_count_train":trainer.negative_count,"negative_positive_ratio_train":trainer.negative_count/trainer.positive_count,"adopted_weight":trainer.parameters.get("class_weight",trainer.parameters.get("scale_pos_weight")),"random_state":config["random_state"],"training_config":trainer.parameters,"library_versions":library_versions(),"metrics":metrics,"dataset_id":dataset_id,"dataset_fingerprint":fingerprints or {},"historical_cache_metadata":hmeta,"feature_audit":audit_result(result),"threshold":.5,"paper_combinations_confirmed":False,"validation_history_semantics":"Frozen through train_end; no validation outcomes" if evaluation_protocol==EvaluationProtocol.RESEARCH_FROZEN else "Rolling strict date < D; prior validation outcomes visible","preprocessor_fit_scope":"TRAIN_ONLY","empty_numeric_columns":trainer.pipeline.named_steps["preprocessor"].empty_numeric_columns_,"warnings":result.warnings+metrics["warnings"],"timings":{"feature_pipeline_seconds":feature_time,**trainer.timings,"validation_predict_seconds":predict_time}}
            bundle=ModelBundle(json_safe(metadata),trainer.pipeline)
            stage("保存模型并登记可信项目资产")
            folder=trainer.save_bundle(bundle,self.manager);saved_identity=identity
            start=time.perf_counter();loaded=self.manager.load_model(identity);load_time=time.perf_counter()-start
            # 重新调用 FeaturePipeline，再经过加载模型进行验证集内部推理。
            smoke=FeaturePipeline(history).build_features(frame.loc[result.X.index[valid[:8]]],combo,mode="INFERENCE",schedule_context=schedule_context,history_visibility_mode=policies["validation_history_policy"],history_cutoff_date=policies["validation_history_cutoff"])
            after=self.manager.predict_proba(loaded,smoke)
            equal=bool(np.allclose(probability[:8],after,rtol=1e-10,atol=1e-12))
            if not equal:raise ValueError("模型保存加载概率不一致")
            experiment.update(status="COMPLETED",finished_at=timestamp(),model_id=identity,metrics=metrics,time_split=split.metadata,parameters=trainer.parameters,serialization_equal=equal,load_seconds=load_time,total_seconds=time.perf_counter()-total)
            atomic_json(experiment_path,json_safe(experiment))
            return {"metadata":bundle.metadata,"model_path":str(folder),"experiment":experiment,"smoke_probabilities":after.tolist(),"serialization_equal":equal,"load_seconds":load_time,"total_seconds":time.perf_counter()-total}
        except Exception as exc:
            if saved_identity is not None:self.manager.registry.archive_model(saved_identity)
            experiment.update(status="FAILED",finished_at=timestamp(),error=str(exc));atomic_json(experiment_path,json_safe(experiment));raise

def train_dataset(context,dataset_id,history_id,schedule_id,combination,model_type,*,formal_requested=False,date_ranges=None,evaluation_protocol=EvaluationProtocol.RESEARCH_FROZEN,progress=None,stage=None,cancelled=None):
    config=load_model_config(context.root);stage=stage or (lambda text:None)
    record=context.datasets.repository.get(dataset_id)
    if record["dataset_type"]!="HISTORICAL_LABELED":raise ValueError("训练需要真实标签列")
    if context.datasets.source_status(record)!="READY":raise ValueError("训练源文件失效，请先重新分析")
    if not record["is_demo"] and not formal_requested:raise ValueError("请明确勾选正式训练授权；DEMO 无需该选项")
    fingerprints={record["file_path"]:file_fingerprint(Path(record["file_path"]))};stage("准备数据（有界内存，保留全部计划行）")
    parts=[];rows=0
    for chunk in FlightDataLoader(record["file_path"],context.config.get("csv_chunksize")).iter_chunks(cancelled=cancelled):
        dates=normalize_frame(chunk).flight_date
        if dates.ge(LOCK_DATE).any():raise ValueError("最终时间外测试集已锁定")
        rows+=len(chunk)
        if rows>config["max_training_rows"]:raise ValueError("训练上限 100000 行，不静默截断或抽样")
        parts.append(chunk)
    if not parts:raise ValueError("数据为空")
    frame=pd.concat(parts,ignore_index=True)
    if progress:progress(20,rows)
    combo=FeatureCombinationRegistry().get_combination(combination);history=None;schedule=None
    if "H" in combo.groups:
        if history_id is None:raise ValueError("请选择已构建的历史缓存")
        history=HistoricalFeatureEngine(context)
        if not history.load_cache(history_id):raise ValueError("历史缓存不存在或失效，请先构建")
    if "C" in combo.groups and schedule_id is not None:
        r=context.datasets.repository.get(schedule_id)
        if context.datasets.source_status(r)!="READY":raise ValueError("计划表源失效")
        fingerprints[r["file_path"]]=file_fingerprint(Path(r["file_path"]))
        def chunks():
            for chunk in FlightDataLoader(r["file_path"],context.config.get("csv_chunksize")).iter_chunks(cancelled=cancelled):
                if normalize_frame(chunk).flight_date.ge(LOCK_DATE).any():raise ValueError("计划上下文涉及锁定最终期")
                yield chunk
        schedule=CongestionFeatureEngineer().fit_chunks(chunks())
    result=TrainingService(context.root).train_frame(frame,model_type,combination,history=history,schedule_context=schedule,is_demo=bool(record["is_demo"]),formal_requested=formal_requested,dataset_id=dataset_id,fingerprints=fingerprints,config=config,date_ranges=date_ranges,stage=stage,evaluation_protocol=evaluation_protocol)
    if progress:progress(100,rows)
    context.logger.info("模型训练完成 model=%s demo=%s",result["metadata"]["model_id"],record["is_demo"])
    return result
