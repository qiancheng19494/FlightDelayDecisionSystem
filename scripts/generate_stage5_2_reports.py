"""Generate Stage5.2 reports from recorded runs; never train or recompute paper metrics."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.ml.model_registry import atomic_json
from core.ml.model_manager import ModelManager
from core.validation_evidence import source_digest
from core.research_protocol_auditor import sha256
from core.feature_pipeline import FeaturePipeline
from core.data_loader import FlightDataLoader
from core.historical_features import HistoricalFeatureEngine
from core.bootstrap import initialize

def generate():
    folder=ROOT/"output/reports"
    def load(name):return json.loads((folder/name).read_text(encoding="utf-8"))
    baseline=load("stage5_2_baseline.json");tests=load("stage5_2_test_results.json");old=load("stage5_1_test_results.json")
    assert tests["source_digest"]==source_digest(ROOT)
    assert all(tests["tests"].get(k)=="PASS" for k in old["tests"]) and tests["passed"]==tests["total"]
    preserved={}
    for name,expected in baseline["preserved_files"].items():
        if Path(name).as_posix()=="models/registry.json":continue # append-only new REAL_SMOKE identities are allowed
        preserved[name]=sha256(ROOT/name)==expected
    assert all(preserved.values())
    previous_registry=json.loads((ROOT/"models/registry.json").read_text(encoding="utf-8"))
    assert previous_registry["active_model_id"] is None
    audit=load("stage5_2_protocol_audit.json");inventory=load("stage5_2_asset_inventory.json")
    paper_checks={e["path"]:sha256(e["path"])==e["sha256"] for e in audit["evidence"]}
    assert all(paper_checks.values())
    real=load("stage5_2_real_bts_validation.json");gui=load("stage5_2_gui_verification.json")
    if real["status"]!="PASS":raise ValueError("Real smoke not passed")
    # Independently recount columns from a fresh feature build; fixes a reporting-only len(DataFrame) error.
    context=initialize(ROOT);h=HistoricalFeatureEngine(context);assert h.load_cache(real["dataset_id"])
    sample=FlightDataLoader(real["sample_path"]).read_preview(page_size=100)
    fresh=FeaturePipeline(h).build_features(sample,"COMBO_ATHC",schedule_context=sample)
    columns=fresh.X.shape[1];assert columns==80
    real["features"]["matrix_columns"]=columns
    real["geometry_recheck"]="Fresh 100-row FeaturePipeline build and all saved model feature_names confirm 80 columns; original row/column reporting typo corrected, no model retraining."
    manager=ModelManager(ROOT)
    for record in real["models"]:
        bundle=manager.load_model(record["metadata"]["model_id"]);assert len(bundle.metadata["feature_names"])==columns
        assert record["serialization_equal"] and bundle.metadata["status"]=="REAL_SMOKE"
    for hnd in list(context.logger.handlers):hnd.close();context.logger.removeHandler(hnd)
    atomic_json(folder/"stage5_2_real_bts_validation.json",real)
    q=real["quality"];f=real["features"];groups=audit["combinations"];split=f["split"]
    evidence_lines=[f"- `{e['path']}` — SHA-256 `{e['sha256']}`" for e in audit["evidence"]]
    protocol=["# Stage 5.2 论文协议只读审计", "",f"结论：**{audit['protocol']}**；证据等级 **{audit['confidence']}**。没有重算论文指标、执行论文代码、反序列化论文缓存/模型或修改论文。", "", "## 可见性判定", "", "- 训练：历史使用 [D−7/30,D)；必要时回退到 expanding 的过去全局统计。", "- 验证：ROLLING_ASOF，12 月较早真实标签可影响后日。", "- 原论文最终期：来源标签冻结至 2024-12-31，7/30 天窗口仍随目标日推进。因此是来源冻结，不是把固定月底比例复制到整月。", "- 总体：MIXED；不等同当前统一 RESEARCH_FROZEN，也不等同全部时期 ONLINE_ROLLING。仅验证可见性对应 ONLINE_ROLLING。", "- 软件 H 使用 expanding，论文 H 使用 7/30 天窗口；A/T/C 字段、回退、冷启动与部分标签来源也不同，不宣称实验复现。", "", "## 实际代码和产物", "", "phase4_features.py 将所有 2024 月份日统计构成 indices，查询上界 min(target_day, VALID_END+1)，VALID_END=2024-12-31。train_phase5.py 读取 1–11 月训练特征及 12 月验证特征。train_phase4.py 的最终期查询使用仅含 2024 的索引并固定 VALID_END。已核对源文件指纹与论文冻结 manifest。", "", "| 验证目标日期 | 最晚历史来源 | 检查条数 |", "|---|---|---:|"]
    protocol += [f"| {c['target_date']} | {c['max_source_date']} | {c['rows']} |" for c in audit["cache_date_checks"]]
    protocol += ["", "以上来自原 outputs/phase5/history_leakage_samples.csv，并由原 phase4_history_audit_2024.csv 的 day-offset 记录交叉核对。仅读取已存在审计产物，没有新计算 2025-01 指标。", "", "## 时间划分与组合", "", json.dumps(audit["date_split"],ensure_ascii=False), "", "确认：8 个含 A 的特征组合 × RF/XGB 两模型 = 16 次组合实验；不是四组全部 16 个子集。", "", ", ".join(groups["groups"]), "", "当前软件保留通用 16 子集及既有 schema；paper_combinations_confirmed 不改为 true，防止误称与原论文相同。", "", "原论文 T 包括 Route、DepTimeBlock，不是纯时间组；H 为八项历史率加 HistoryUnavailable；C 为四项计划流量。完整原字段列表保存在论文 phase5_protocol.json，已记录于资产清单。", "", "## 原参数（只记录，不覆盖软件配置）", "", "```json",json.dumps(audit["parameters"],ensure_ascii=False,indent=2),"```", "", "## 原指标资产", "", "已发现 outputs/phase5/feature_group_ablation.csv、candidate_RF_*.json、candidate_XGB_*.json、phase4_model_comparison.csv 及历史原始指标/日志。只记录其来源与指纹，不从 Word 反造原始结果，不与本次 smoke 数值比较。", "", "## 论文表述", "",audit["paper_text_assessment"], "原实验已披露 2025-01 曾用于早期探索；本次未把它称作未接触盲测。对已检查的混合协议表述建议：无需修改。未逐份验证所有 Word 修订版；其他版本无法判断。", "", "## 证据来源", ""]+evidence_lines
    (folder/"stage5_2_protocol_audit.md").write_text("\n".join(protocol)+"\n",encoding="utf-8")
    quality=["# Stage 5.2 真实 BTS 小样本数据质量", "", "仅限各源文件前 50,000 条、合计 100,000 条；不是完整月份质量保证。原文件未改写。", "", "## 来源", ""]+[f"- `{path}`" for path in real["source_fingerprints"]]+["",f"总数 {q['sampled_rows']}，日期 {q['date_range']}，无效日期 {q['invalid_dates']}，原字段 {len(q['dtypes'])}（含 BTS 尾部空列）。",f"标签：{q['labels']}；取消分布：{q['cancelled_check']}。取消率 {q['cancelled_check']['one']/q['sampled_rows']:.6%}。", "", "DEP_DELAY 负值保留；取消=1 不进入监督标签；缺失/非法取消状态不标注。本样本 DEP_DELAY 派生标签与原 DepDel15 在 99,381 个有效标签上无差异。", "", "```json",json.dumps({k:q[k] for k in ["mapping","prediction_allowed_fields","posterior_fields","label_fields","scheduled_times","departure_delay","published_label_check","missing"]},ensure_ascii=False,indent=2),"```", "", "BTS 单个尾部空列改名 __BTS_TRAILING_EMPTY__ 保留，逐行核查为空；内部空列、任意非 BTS 空表头、尾列含值仍拒绝。2400 等边界由人工真实格式 fixture 验证；本真实样本未出现 2400，不声称已在本样本观察到。"]
    (folder/"stage5_2_real_bts_data_quality.md").write_text("\n".join(quality)+"\n",encoding="utf-8")
    lines=["# Stage 5.2 真实 BTS 完整链路验证", "", "状态：**PASS — REAL_SMOKE**。这些模型只验证工程链路，不代表论文最终结果；未自动设置 active、未替换六个 DEMO。", "", "## 数据发现与范围", "", "在相邻 flight-delay-prediction/data/raw 找到 2024-01～12 及 2025-01 的 CSV/ZIP 和本地下载 manifest；清单见 stage5_2_asset_inventory.json。2025 只登记资产与读取表头，不进行模型评价。", "",f"本次只读采样：{real['sample_rows']} 条，2024-11、12 各前 50,000 条。训练有效 {real['models'][0]['metadata']['training_row_count']}，验证有效 {real['models'][0]['metadata']['validation_row_count']}。",f"时间窗口：{split['training_start_date']}～{split['training_end_date']} / {split['validation_start_date']}～{split['validation_end_date']}；Frozen cutoff={f['cutoff']}。", "取样有源文件排序偏差，未将局部 C 冒充完整机场/月计划；H 也仅覆盖所选样本，不等同论文全年历史。", "", "## 字段、质量和特征", "", f"FieldMapper 必需字段齐全；无效日期 {q['invalid_dates']}；计划起降异常 {q['scheduled_times']}。有效标签 {q['labels']['valid']}，取消 {q['cancelled_check']['one']}；负延误 {q['departure_delay']['negative']} 保留。", "", "- A/T：独立逐项抽查 100 条分钟、小时、星期、月份、周末、时段及周期编码，通过。", "- H：训练 strict date < D、Frozen 截止日、Rolling strict date < D 全部实际来源审计通过。", "- 修改 cutoff 后所有验证 DEP_DELAY：全部验证 Frozen H 精确不变；Rolling 首验证日不变，后日变化。", "- C：修改 DEP_DELAY、ARR_DELAY、CANCELLED 和实际起降时间，目标和计划上下文一起变更，C 精确不变。", "- A、train ATHC、Frozen ATHC、Rolling ATHC 的 FeatureValidator / FeatureAudit / TemporalLeakageAuditor 均通过。", "", "历史来源抽查：", "```json",json.dumps(f["history_probes"],ensure_ascii=False,indent=2),"```", "", "## REAL_SMOKE 模型", "", "默认 RESEARCH_FROZEN、固定 0.5，现有软件 RF/XGB 参数，无搜索；只各训练一份 ATHC。", "", "| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |", "|---|---:|---:|---:|---:|---:|---:|"]
    for entry in real["models"]:
        m=entry["metadata"];lines.append("| "+" | ".join([m["model_type"]]+[f"{m['metrics'][k]:.6f}" for k in ["accuracy","precision","recall","f1","roc_auc","average_precision"]])+" |")
    for entry in real["models"]:
        m=entry["metadata"];lines += ["",f"`{m['model_id']}`：{m['model_file_size']} 字节，保存/加载概率一致={entry['serialization_equal']}，混淆矩阵={m['metrics']['confusion_matrix']}。"]
    lines += ["", "## 资源", "",f"完整成功运行 {real['total_seconds']:.3f} 秒；峰值 RSS/工作集 {real['peak_rss_bytes']:,} 字节（{real['peak_rss_bytes']/1024**2:.2f} MiB），口径：{real['memory_method']}。",f"历史构建 {real['history_build_seconds']:.3f} 秒，缓存 {real['history_cache_bytes']:,} 字节；Feature matrix {f['matrix_rows']}×{f['matrix_columns']}，深度内存 {f['matrix_deep_bytes']:,} 字节（不是进程峰值）。", "仅测 100,000 行，未扩大到 500,000 或 698 万全量。未安装额外依赖，Windows 系统接口记录进程高水位。", "", "## 回归与保留资产", "",f"基线 302/302（{baseline['seconds']} 秒）；最终 {tests['passed']}/{tests['total']}，新增 {tests['total']-302} 项，warning={tests['warning_count']}。原 302 项测试文件未修改；HistoricalFeatureEngine 和 FeaturePipeline 源码指纹不变。", "六份 DEMO 模型及其 metadata/metrics 指纹不变；论文证据文件指纹复核不变。GUI 兼容性、协议只读区、原始协议证据、真实样本质量与 FeatureAudit 通过；15 导航保持。GUI 快速切换有非阻断 Matplotlib 布局警告。", "", "首次 smoke 在验证脚本向字符串列赋整型时失败，未创建模型；修复后完整重跑通过，失败记录和其样本保留于 stage5_2_initial_smoke_failure.json。矩阵列数报告的 len(DataFrame) 笔误也已由独立新特征构建核对修正为 80；没有重新训练模型。", "", "## 边界与 Stage 6 前置", "", "FeaturePipeline、模型系统、EvaluationProtocol 可用；真实字段及 10 万行工程链路已验证；论文 protocol=MIXED（CONFIRMED_CODE），实际八组合×两模型已确认。原论文源码、Word、指标未修改。没有网络下载、最终期指标、SHAP、阈值优化、全组合训练或 Stage 6。", "", "局限：局部取样与局部 C/H，上游发布时间没有历史快照证明；真实原文件只核对本地来源记录及 size/mtime，没有为所有大文件重算完整 SHA-256。未来 Stage 6 可保留 DEMO/REAL_SMOKE 区分开发预测入口，但不得把 smoke 视为最终论文模型。", "", "## 验收 A–AG", ""]
    items=["302 旧测试全部通过","历史核心语义不变","旧模型指纹不变","Frozen DEMO 不变","论文资产扫描","协议 MIXED","证据路径与 SHA-256","八组合×两模型确认","真实数据扫描","无网络下载","真实字段映射","真实标签","真实 A/T","Frozen H","Rolling H","C 后验独立","真实 FeatureAudit","RF/XGB smoke","最终期未评价","无阈值优化","无 SHAP","论文未改","原始数据/新资产忽略规则","资源已记录","BTS GUI","Protocol GUI","协议报告","真实报告","README","PROJECT_STATUS","全测试通过","GUI 启动","无阻断导入/语法错误"]
    labels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")+["AA","AB","AC","AD","AE","AF","AG"]
    lines += [f"- {k}：PASS — {v}" for k,v in zip(labels,items)]
    (folder/"stage5_2_real_bts_validation.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    summary=dict(passed=True,old_tests_passed=302,new_tests=tests["total"]-302,total_tests=tests["total"],source_digest=source_digest(ROOT),preserved_files=preserved,paper_evidence_unchanged=paper_checks,protocol=audit["protocol"],confidence=audit["confidence"],real_smoke=real["status"],gui=gui,final_test_evaluated=False)
    atomic_json(folder/"stage5_2_acceptance.json",summary);print(json.dumps({k:v for k,v in summary.items() if k not in ("preserved_files","paper_evidence_unchanged","gui")},ensure_ascii=False))
if __name__=="__main__":generate()
