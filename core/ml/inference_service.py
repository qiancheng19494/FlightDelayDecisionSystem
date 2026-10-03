"""Stage 5 内部推理能力；保留 DEMO 身份，尚未接单航班预测 GUI。"""
from dataclasses import dataclass
import numpy as np
from core.feature_pipeline import FeaturePipeline
from core.history_normalization import normalize_frame
from core.historical_features import HistoricalFeatureEngine
from core.ml.model_manager import ModelManager
from core.ml.time_split import LOCK_DATE
from core.history_visibility import EvaluationProtocol,history_description
@dataclass
class PredictionResult:
    model_id: str
    model_type: str
    combination_id: str
    probability_delay: list
    probability_normal: list
    threshold: float
    predicted_label: list
    feature_schema_version: int
    warnings: list
    is_demo: bool
class InferenceService:
    def __init__(self,context):self.context=context;self.manager=ModelManager(context.root)
    def predict(self,frame,*,model_id=None,schedule_context=None,evaluation_protocol=None):
        dates=normalize_frame(frame).flight_date
        if dates.isna().any():raise ValueError("推理日期无效，无法检查最终期锁定")
        if dates.ge(LOCK_DATE).any():raise ValueError("Stage 5 最终时间外测试集锁定，禁止内部预测")
        identity=model_id or self.manager.registry.read()["active_model_id"]
        if identity is None:raise ValueError("尚未设置当前模型")
        bundle=self.manager.load_model(identity);history=None
        if "H" in bundle.metadata["feature_groups"]:
            stored=bundle.metadata["historical_cache_metadata"];history=HistoricalFeatureEngine(self.context)
            if not history.load_cache(stored["dataset_id"]):raise ValueError("模型所需历史缓存不可用")
            if history.config!=stored["configuration"] or history.folder.name!=stored["cache_id"]:raise ValueError("历史缓存/配置与模型资产不一致，请明确重建模型")
        selected=EvaluationProtocol(evaluation_protocol or bundle.metadata.get("evaluation_protocol","ONLINE_ROLLING"))
        visibility="FROZEN_CUTOFF" if selected==EvaluationProtocol.RESEARCH_FROZEN else "ROLLING_ASOF"
        cutoff=bundle.metadata["training_end_date"] if visibility=="FROZEN_CUTOFF" else None
        result=FeaturePipeline(history).build_features(frame,bundle.metadata["combination_id"],mode="INFERENCE",schedule_context=schedule_context,history_visibility_mode=visibility,history_cutoff_date=cutoff)
        result.warnings.append(f"Inference protocol: {selected.value}; cutoff: {cutoff}; " + history_description(bundle.metadata))
        p=self.manager.predict_proba(bundle,result,history_policy_override=dict(history_visibility_mode=visibility,history_cutoff_date=cutoff) if evaluation_protocol is not None else None)
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError("模型输出概率非法")
        return PredictionResult(identity,bundle.metadata["model_type"],bundle.metadata["combination_id"],p.tolist(),(1-p).tolist(),.5,(p>=.5).astype(int).tolist(),bundle.metadata["feature_schema_version"],result.warnings,bundle.metadata["is_demo"])
