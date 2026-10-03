"""执行实际特征审计并引用带源码指纹的全量测试证据。"""
import json,sys
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from pandas.testing import assert_frame_equal
from core.feature_groups import *
from core.historical_features import HistoricalFeatureEngine,json_safe
from core.feature_pipeline import FeaturePipeline
from core.congestion_features import CongestionFeatureEngineer
from core.feature_service import audit_result,export_feature_sample
from core.validation_evidence import source_digest
from core.bootstrap import initialize

def generate():
    frame=pd.read_csv(ROOT/"data/demo/demo_schedule_flights.csv",keep_default_na=False)
    context=initialize(ROOT)
    record=next((r for r in context.datasets.datasets() if Path(r["file_path"])==ROOT/"data/demo/demo_schedule_flights.csv"),None)
    if record is None:raise ValueError("请先在数据管理中导入 demo_schedule_flights.csv 并构建历史缓存")
    history=HistoricalFeatureEngine(context)
    if not history.load_cache(record["id"]):raise ValueError("请先为计划 DEMO 构建有效历史缓存")
    pipeline=FeaturePipeline(history);r=pipeline.build_features(frame,mode="TRAIN",schedule_context=frame,dataset_id=record["id"])
    repeated=pipeline.build_features(frame,mode="TRAIN",schedule_context=frame)
    deterministic=r.X.equals(repeated.X)
    combinations=[]
    for c in FeatureCombinationRegistry().list_combinations():
        actual=pipeline.build_features(frame,c,schedule_context=frame)
        combinations.append({**asdict(c),"passed":actual.metadata["validation"]["passed"]})
    c=CongestionFeatureEngineer().fit(frame).transform(frame);independence={}
    for field in ["DEP_DELAY","ARR_DELAY","CANCELLED","DEP_TIME","ARR_TIME"]:
        mutated=frame.copy();mutated[field]=99999
        independence[field]=c.equals(CongestionFeatureEngineer().fit(mutated).transform(mutated))
    audit=audit_result(r)
    evidence_path=ROOT/"output/reports/stage4_test_results.json"
    evidence=json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}
    evidence_current=evidence.get("source_digest")==source_digest(ROOT)
    benchmark_path=ROOT/"output/reports/stage4_benchmark.json"
    benchmark=json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else None
    report={"feature_schema_version":1,"group_fields":GROUP_FIELDS,"group_schema":GROUP_SCHEMA,"lineage":r.metadata["lineage"],"combinations":combinations,"paper_combinations_confirmed":False,"feature_audit":audit,"H_audit":r.metadata["history_audit"],"C_outcome_independence":independence,"deterministic":deterministic,"test_evidence_current":evidence_current,"tests":evidence,"benchmark":benchmark,"warnings":r.warnings,"limits":["真实 BTS 尚待实测","计划表完整性和预测时可获得性由调用方确认；不冒充实时流量","按 flight_date 航班批次分组；到达跨日不推断到达服务日期/时区","未实现可选 ±30 分钟窗口","峰值进程内存未测量"]}
    report["passed"]=audit["passed"] and deterministic and all(independence.values()) and all(c["passed"] for c in combinations) and evidence_current and evidence.get("failed")==0 and evidence.get("errors")==0 and benchmark is not None
    exported=export_feature_sample(r,ROOT/"output/features");report["sample_export"]=str(exported)
    text=["# 第四阶段特征验证报告","",f"Feature schema version：1。实际综合审计：{'PASS' if report['passed'] else '未通过 / 证据缺失'}。","","仅生成特征；未训练模型、评价模型效果或运行论文实验。论文实际组合尚待确认。","","## A/T/H/C 字段"]
    for g,fields in GROUP_FIELDS.items():text.extend(["",f"### {g}（{len(fields)}）","",", ".join(fields)])
    text.extend(["","## 16 组合","","| ID | 组 | 字段数 | 实际检查 |","| --- | --- | ---: | --- |"])
    for c in combinations:text.append(f"| {c['combination_id']} | {c['name']} | {len(c['feature_names'])} | {'PASS' if c['passed'] else 'FAIL'} |")
    text.extend(["","## 特征审计","","| 检查 | 结果 |","| --- | --- |"])
    for check in audit["checks"]:text.append(f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} |")
    for field,passed in independence.items():text.append(f"| 修改 {field} 不改变 C | {'PASS' if passed else 'FAIL'} |")
    text.extend([f"| 相同输入输出确定 | {'PASS' if deterministic else 'FAIL'} |","","H 通过现有 HistoricalFeatureEngine 1.0.1 查询，独立运行来源日期审计；没有修改其缓存逻辑。","","## T 边界与测试证据",f"当前源码匹配：{evidence_current}。总测试 {evidence.get('total')}，通过 {evidence.get('passed')}，warning {evidence.get('warning_count')}。",""])
    for name,status in evidence.get("tests",{}).items():
        if any(key in name for key in ["TemporalTests","test_current_H_unchanged_future_H_changed","test_all_sixteen_unions"]):text.append(f"- {name}: {status}")
    text.extend(["","## Benchmark","","```json",json.dumps(benchmark,ensure_ascii=False,indent=2),"```","","## 限制",*['- '+v for v in report['limits']],"","X/y 样本："+str(exported)])
    (ROOT/"output/reports/stage4_feature_validation.json").write_text(json.dumps(json_safe(report),ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"output/reports/stage4_feature_validation.md").write_text("\n".join(text),encoding="utf-8")
    print(json.dumps({"passed":report["passed"],"tests_current":evidence_current,"sample_export":str(exported)}))
    return report
if __name__=="__main__":raise SystemExit(0 if generate()["passed"] else 1)
