"""从真实审计与绑定源码的测试证据生成 JSON / Markdown 历史报告。"""
import json
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4
from core.historical_features import json_safe
from core.validation_evidence import load_test_evidence

KEY_TESTS={"test_same_day_label_mutation":"修改当天标签", "test_future_labels_mutation":"修改未来标签", "test_append_future_month":"追加未来月份", "test_shuffled_rows_exactly_equal":"随机打乱行序", "test_directional_route":"路线方向", "test_weighted_average_not_daily_average":"加权均值", "test_cold_route_keeps_zero_count":"回退不改变实体数量"}

def report_data(engine) -> dict:
    engine._assert_fresh()
    return json_safe({"software":"航班运行数据智能分析与延误预测系统 V1.0","generated_at":datetime.now(timezone.utc).isoformat(),"metadata":engine.metadata,"validation":engine.metadata["validation"],"test_evidence":load_test_evidence(engine.root),"disclaimer":"历史描述性统计，不代表因果关系或航空安全评价；DEMO 非论文数据。"})

def markdown_report(data: dict) -> str:
    m,v=data["metadata"],data["validation"]
    lines=[f"# {data['software']} — 历史特征验证报告", "", f"生成时间：{data['generated_at']}", "", f"数据集：{m.get('dataset_name',m['source_file'])}（ID {m['dataset_id']}）", "", f"时间范围：{m['source_min_date']} 至 {m['source_max_date']}；窗口：{m['history_window_mode']}；schema：{m['historical_feature_schema_version']}；引擎：{m['engine_version']}；min_entity_history：{m['min_entity_history']}。", "", "规则：所有历史来源严格 date < target_date。同日和未来标签不参与；实体数量始终保留原值，回退只作用于有效率与均值。", "", "## 实体规模与冷启动", "", "| 实体 | 实体数 | 日聚合行数 | 冷启动目标数 |", "| --- | ---: | ---: | ---: |"]
    for e in m["entity_types"]:lines.append(f"| {e} | {m['entity_counts'][e]} | {m['aggregate_rows'][e]} | {v['cold_start_counts'][e]} |")
    lines += ["", "## 回退来源次数", "", "```json", json.dumps(v["fallback_source_frequency"],ensure_ascii=False,indent=2), "```", "", "## 自动时间审计", "", f"结果：{'PASS' if v['audit']['passed'] else 'FAIL'}；范围：{v['audit']['mode']}；实际检查 {v['audit']['rows_checked']} 行。", "", "| 检查项目 | 状态 | 说明 |", "| --- | --- | --- |"]
    for item in v["audit"].get("check_results",[]):
        lines.append(f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | {item['description']} |")
    lines += ["", "## 关键自动测试", ""]
    evidence=data["test_evidence"]
    if evidence.get("status")=="COMPLETED":
        lines.append(f"总数 {evidence['total']}；通过 {evidence['passed']}；失败 {evidence['failed']}；错误 {evidence['errors']}；跳过 {evidence['skipped']}；warning {evidence['warning_count']}。")
        lines += ["", "| 变形/边界测试 | 执行结果 |", "| --- | --- |"]
        for suffix,title in KEY_TESTS.items():
            results=[value for name,value in evidence["tests"].items() if name.endswith('.'+suffix)]
            lines.append(f"| {title} | {', '.join(results) if results else 'NOT_RUN'} |")
    else:lines.append(evidence.get("description","尚未记录测试结果"))
    lines += ["", "## 缓存与数据范围", "", f"源文件：{m['source_file']}", "", f"大小：{m['source_file_size']} 字节；修改时间 ns：{m['source_modified_time']}；缓存生成：{m['generated_at']}。", "", f"源记录 {m['source_rows']}；有效历史 {m.get('valid_history_rows',0)}；首日无历史 {v['first_date_without_history']}；无效目标日期显式跳过 {v['invalid_target_dates_skipped']}。", "", "## 抽样查询", ""]
    for sample in v["sample_queries"]:
        target=sample["target"];features=sample["features"]
        lines.append(f"- {target['flight_date']} / {target['airline']} / {target['origin']} → {target['destination']}：global_count={features['hist_global_count']}，route_count={features['hist_route_count']}，route_rate={features['hist_route_delay_rate']}，来源={sample['metadata']['hist_route_fallback_source']}。")
    lines += ["", "## 特征 schema", "", ", ".join(m["feature_schema"]), "", "## 警告与适用范围", ""]
    lines += m["warnings"] or ["本数据集未发现历史引擎警告。"]
    lines += ["",data["disclaimer"],"当前自动审计检查生成结果及其来源记录；不能替代真实 BTS 数据质量检查，也不声称对任意外部篡改绝对安全。", ""]
    return "\n".join(lines)

def export_history_report(engine,path: Path) -> str:
    data=report_data(engine)
    path=Path(path)
    if not path.suffix:path=path.with_suffix(".md")
    content=markdown_report(data) if path.suffix.lower()==".md" else json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)
    temporary=path.with_name(path.name+"."+uuid4().hex+".tmp")
    try:
        temporary.write_text(content,encoding="utf-8")
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)
    return str(path)
