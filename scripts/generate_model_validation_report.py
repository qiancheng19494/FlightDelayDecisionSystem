"""引用真实训练资产，重新核验兼容、预处理训练范围和内部推理。"""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from core.bootstrap import initialize
from core.historical_features import HistoricalFeatureEngine,json_safe
from core.feature_pipeline import FeaturePipeline
from core.ml.model_manager import ModelManager,library_versions
from core.ml.training_service import load_model_config
from core.ml.time_split import TemporalDataSplitter
from core.ml.inference_service import InferenceService
from core.validation_evidence import source_digest

def generate():
    context=initialize(ROOT);manager=ModelManager(ROOT);evidence=json.loads((ROOT/"output/reports/stage5_test_results.json").read_text(encoding="utf-8"))
    current=evidence["source_digest"]==source_digest(ROOT);models=[]
    for entry in manager.registry.list_models():
        if entry["status"]!="DEMO" or entry["dataset_id"]!=2:continue
        bundle=manager.load_model(entry["model_id"]);m=bundle.metadata
        record=context.datasets.repository.get(m["dataset_id"]);frame=pd.read_csv(record["file_path"],keep_default_na=False);history=None
        if "H" in m["feature_groups"]:
            history=HistoricalFeatureEngine(context)
            if not history.load_cache(m["historical_cache_metadata"]["dataset_id"]):raise ValueError("历史缓存失效")
        result=FeaturePipeline(history).build_features(frame,m["combination_id"],mode="TRAIN",schedule_context=frame)
        split=TemporalDataSplitter().split(pd.to_datetime(frame.FL_DATE),is_demo=True)
        train=split.train_positions[result.y.iloc[split.train_positions].notna().to_numpy()];valid=split.validation_positions[result.y.iloc[split.validation_positions].notna().to_numpy()]
        pre=bundle.preprocessor;numeric=pre.numeric_columns_;expected=result.X.iloc[train][numeric].median().fillna(0).to_numpy(dtype=float)
        median_ok=bool(np.allclose(expected,pre.transformer_.named_transformers_["numeric"].statistics_,rtol=0,atol=1e-12))
        cat_ok=True
        if pre.categorical_columns_:
            categories=pre.transformer_.named_transformers_["categorical"]["onehot"].categories_
            for name,values in zip(pre.categorical_columns_,categories):
                expected_categories=set(result.X.iloc[train][name].astype(object).where(result.X.iloc[train][name].notna(),"__MISSING__"))
                cat_ok &= set(values)==expected_categories
        smoke=InferenceService(context).predict(frame.iloc[valid[:8]],model_id=m["model_id"],schedule_context=frame)
        again=manager.load_model(m["model_id"]);sample=FeaturePipeline(history).build_features(frame.iloc[valid[:8]],m["combination_id"],schedule_context=frame)
        consistent=bool(np.allclose(smoke.probability_delay,manager.predict_proba(again,sample),rtol=1e-10,atol=1e-12))
        rejects={}
        for key,value in [("feature_schema_version",999),("feature_names",["departure_delay"]),("combination_id","COMBO_EMPTY")]+([("historical_schema_version",999)] if "H" in m["feature_groups"] else []):
            wrong=copy.deepcopy(m);wrong[key]=value
            try:manager.validate_metadata(wrong);rejects[key]=False
            except ValueError:rejects[key]=True
        models.append({"metadata":m,"checks":{"train_only_imputer":median_ok,"train_only_encoder":cat_ok,"reload_prediction_equal":consistent,"probabilities_in_range":all(0<=p<=1 for p in smoke.probability_delay),"schema_rejections":rejects},"smoke_probabilities":smoke.probability_delay})
    benchmark=json.loads((ROOT/"output/reports/stage5_benchmark.json").read_text(encoding="utf-8"));gui=json.loads((ROOT/"output/reports/stage5_gui_verification.json").read_text(encoding="utf-8"))
    passed=current and evidence["failed"]==0 and evidence["errors"]==0 and len(models)==4 and all(all(v for k,v in m["checks"].items() if k!="schema_rejections") and all(m["checks"]["schema_rejections"].values()) for m in models)
    report={"software":"航班运行数据智能分析与延误预测系统 V1.0","stage":5,"passed":passed,"library_versions":library_versions(),"config":load_model_config(ROOT),"feature_schema":1,"historical_schema":1,"models":models,"tests":evidence,"test_evidence_current":current,"benchmark":benchmark,"gui":gui,"final_test_locked":True,"threshold":.5,"threshold_optimization":False,"paper_combinations_confirmed":False,"real_BTS_verified":False,"statement":"本阶段模型仅用于软件工程与训练链路验证，不用于与论文实验结果比较。","history_semantics":"H 为逐日 as-of；先前验证日结果可用于更晚验证日历史，不参与预处理或估计器训练。"}
    text=["# 第五阶段模型验证报告","",report["statement"],"",f"综合核验：{'PASS' if passed else 'FAIL'}；源码证据匹配：{current}。","","Feature schema=1，History schema=1，历史引擎=1.0.1。threshold 固定 0.5，最终时间外集锁定，未执行 SHAP 或阈值搜索。","","## 依赖与参数","","```json",json.dumps({"versions":report["library_versions"],"config":report["config"]},ensure_ascii=False,indent=2),"```","","## DEMO VALIDATION METRICS","","| 模型 | 组合 | train/validation | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |","| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in models:
        m=r["metadata"];v=m["metrics"];text.append(f"| {m['model_type']} | {m['combination_id']} | {m['training_row_count']}/{m['validation_row_count']} | "+" | ".join(f"{v[k]:.6f}" if v[k] is not None else "N/A" for k in ["accuracy","precision","recall","f1","roc_auc","average_precision"])+" |")
    for r in models:
        m=r["metadata"];text.extend(["",f"### {m['model_id']}","",f"DEMO MODEL。训练 {m['training_start_date']}～{m['training_end_date']}；验证 {m['validation_start_date']}～{m['validation_end_date']}。",f"正类率：训练 {m['positive_rate_train']:.6f}，验证 {m['positive_rate_validation']:.6f}；文件 {m['model_file_size']} 字节。",f"混淆矩阵 [[TN,FP],[FN,TP]]：{m['metrics']['confusion_matrix']}。","","检查："+json.dumps(r["checks"],ensure_ascii=False),"","空数值列："+str(m["empty_numeric_columns"]),"","路径：models/bundles/"+m["model_id"]])
    text.extend(["","## 测试与审计",f"总数 {evidence['total']}，通过 {evidence['passed']}，warning {evidence['warning_count']}。",report["history_semantics"],"验证标签不参与 imputer/encoder/estimator fit；训练/验证按完整日期隔离。C 保留计划取消航班，不使用结果字段。全部原始特征经 FeaturePipeline 和 FeatureValidator。","","## Benchmark","","```json",json.dumps(benchmark,ensure_ascii=False,indent=2),"```","","## 限制","真实 BTS 尚待实测。模型均为 DEMO，通用 16 子集仍待论文定义确认。训练有界内存上限 100000 行，峰值内存未测量。原计划表部分字段缺失，训练期全空数值列保持列位并填 0；不代表真实数据建模方案已确定。","joblib/pickle 仅可信项目资产；兼容性和 SHA-256 检查不是对恶意本地注册表的安全签名。依赖版本目前精确匹配，升级需重新验证。","模型管理默认不删除，未自动挑选或激活最佳模型；Stage 6 尚未开始。"])
    atomic=ROOT/"output/reports/stage5_model_validation.json";atomic.write_text(json.dumps(json_safe(report),ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"output/reports/stage5_model_validation.md").write_text("\n".join(text),encoding="utf-8")
    print(json.dumps({"passed":passed,"models":len(models),"tests":evidence["total"],"current":current}))
    return report
if __name__=="__main__":raise SystemExit(0 if generate()["passed"] else 1)
