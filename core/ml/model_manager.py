"""只加载本项目登记、哈希一致且 schema 兼容的可信资产。"""
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
import json,time
import joblib
from sklearn.pipeline import Pipeline
from core.feature_groups import FeatureCombinationRegistry,FEATURE_SCHEMA_VERSION
from core.history_schema import SCHEMA_VERSION
from core.feature_validator import FeatureValidator
from core.history_cache import digest
from core.ml.model_registry import ModelRegistry,atomic_json,validate_id
from core.ml.model_bundle import ModelBundle
from core.ml.preprocessor import ModelPreprocessor
from core.history_visibility import protocol_metadata

def library_versions():
    names=["scikit-learn","xgboost","joblib","numpy","pandas","scipy"]
    out={}
    for n in names:
        try:out[n]=version(n)
        except PackageNotFoundError:out[n]="unavailable"
    return out
class ModelManager:
    def __init__(self,root):self.root=Path(root);self.registry=ModelRegistry(root)
    def validate_metadata(self,m):
        required={"model_id","model_name","model_type","model_version","is_demo","created_at","combination_id","feature_schema_version","historical_schema_version","feature_names","feature_groups","training_start_date","training_end_date","validation_start_date","validation_end_date","training_row_count","validation_row_count","positive_rate_train","positive_rate_validation","random_state","training_config","library_versions","metrics","dataset_fingerprint","historical_cache_metadata","status","final_test_locked","threshold"}
        if not required.issubset(m):raise ValueError("模型与当前特征架构不兼容：metadata 不完整")
        if m["model_version"] >= 2 or "evaluation_protocol" in m:
            try:
                policies=protocol_metadata(m["evaluation_protocol"],m["training_end_date"],m["validation_end_date"])
                if any(k not in m or m[k]!=v for k,v in policies.items()):raise ValueError("history policy mismatch")
            except (KeyError,TypeError,ValueError) as exc:raise ValueError("模型历史评价协议或 cutoff 不完整/不兼容") from exc
        c=FeatureCombinationRegistry().validate_combination(m["combination_id"])
        valid=m["model_type"] in ("rf","xgboost") and m["feature_schema_version"]==FEATURE_SCHEMA_VERSION and m["feature_names"]==list(c.feature_names) and m["feature_groups"]==list(c.groups) and len(c.feature_names)>0
        valid=valid and m["threshold"]==.5 and m["final_test_locked"] is True and type(m["is_demo"]) is bool
        valid=valid and ((m["is_demo"] and m["status"]=="DEMO") or (not m["is_demo"] and (m["status"]=="TRAINED" or (m["status"]=="REAL_SMOKE" and m.get("training_purpose")=="REAL_SMOKE" and bool(m.get("dataset_fingerprint"))))))
        valid=valid and (m["historical_schema_version"]==SCHEMA_VERSION if "H" in c.groups else m["historical_schema_version"]=="not_required")
        import pandas as pd
        try:
            a,b,c,d=[pd.Timestamp(m[k]) for k in ("training_start_date","training_end_date","validation_start_date","validation_end_date")]
            valid=valid and a<=b<c<=d<pd.Timestamp("2025-01-01") and m["training_row_count"]>0 and m["validation_row_count"]>0
        except (ValueError,TypeError):valid=False
        for name,current in library_versions().items():
            if name=="xgboost" and m["model_type"]!="xgboost":continue
            valid=valid and m["library_versions"].get(name)==current and current!="unavailable"
        if not valid:raise ValueError("模型与当前特征架构不兼容。")
        return c
    def save_bundle(self,bundle):
        m=dict(bundle.metadata);self.validate_metadata(m);identity=validate_id(m["model_id"])
        folder=self.root/"models/bundles"/identity;folder.mkdir(parents=True,exist_ok=False)
        start=time.perf_counter()
        try:
            joblib.dump(bundle.pipeline,folder/"model.joblib",compress=3)
            m["model_sha256"]=digest(folder/"model.joblib");m["model_file_size"]=(folder/"model.joblib").stat().st_size
            m["save_seconds"]=time.perf_counter()-start
            atomic_json(folder/"metrics.json",m["metrics"]);atomic_json(folder/"metadata.json",m)
            entry={**m,"metadata_sha256":digest(folder/"metadata.json"),"metrics_sha256":digest(folder/"metrics.json"),"trusted_project_asset":True}
            self.registry.register_model(entry);bundle.metadata=m
        except Exception:
            (folder/"INCOMPLETE.txt").write_text("保存/登记失败，不可自动加载。",encoding="utf-8");raise
        return folder
    def load_model(self,identity):
        entry=self.registry.get_model(identity)
        if entry.get("status")=="ARCHIVED":raise ValueError("模型已归档")
        if entry.get("trusted_project_asset") is not True:raise ValueError("UNVERIFIED_MODEL_ASSET：来源未确认")
        folder=self.root/"models/bundles"/validate_id(identity)
        if folder.resolve().parent!=(self.root/"models/bundles").resolve():raise ValueError("资产路径越界")
        if (folder/"INCOMPLETE.txt").exists():raise ValueError("模型资产不完整")
        for name,field in [("metadata.json","metadata_sha256"),("metrics.json","metrics_sha256"),("model.joblib","model_sha256")]:
            if not (folder/name).is_file() or digest(folder/name)!=entry.get(field):raise ValueError("模型文件缺失或指纹不匹配")
        metadata=json.loads((folder/"metadata.json").read_text(encoding="utf-8"));self.validate_metadata(metadata)
        if metadata["model_id"]!=identity:raise ValueError("模型 ID 不匹配")
        # 以上检查均发生在反序列化之前；注册表和本地目录必须可信。
        pipeline=joblib.load(folder/"model.joblib")
        if not isinstance(pipeline,Pipeline) or list(pipeline.named_steps)!=["preprocessor","estimator"] or not isinstance(pipeline.named_steps["preprocessor"],ModelPreprocessor):raise ValueError("模型缺少联合预处理器")
        if list(pipeline.named_steps["preprocessor"].feature_names)!=metadata["feature_names"]:raise ValueError("预处理字段不匹配")
        if not hasattr(pipeline.named_steps["preprocessor"],"transformer_"):raise ValueError("预处理器未拟合")
        estimator=pipeline.named_steps["estimator"]
        if type(estimator).__name__!={"rf":"RandomForestClassifier","xgboost":"XGBClassifier"}[metadata["model_type"]]:raise ValueError("估计器类型不匹配")
        if list(estimator.classes_)!=[0,1]:raise ValueError("分类标签 schema 不匹配")
        return ModelBundle(metadata,pipeline)
    def predict_proba(self,bundle,result,*,history_policy_override=None):
        if result.metadata["combination_id"]!=bundle.metadata["combination_id"]:raise ValueError("模型与输入 Combination 不匹配")
        self.validate_metadata(bundle.metadata)
        if "H" in bundle.metadata["feature_groups"]:
            expected=history_policy_override or dict(history_visibility_mode=bundle.metadata.get("history_visibility_mode","ROLLING_ASOF"),history_cutoff_date=bundle.metadata.get("history_cutoff_date"))
            if any(result.metadata.get(k)!=v for k,v in expected.items()):raise ValueError("输入 H 历史可见性协议/cutoff 与模型不一致；请显式选择推理协议")
        FeatureValidator().validate(result.X,bundle.metadata["combination_id"],lineage=result.metadata["lineage"],schema_version=result.metadata["feature_schema_version"])
        from core.feature_service import audit_result
        if not audit_result(result)["passed"]:raise ValueError("输入特征审计失败")
        return bundle.pipeline.predict_proba(result.X)[:,1]
