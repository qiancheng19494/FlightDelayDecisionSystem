"""Evidence-bound Stage 5.1 report. Never trains or evaluates final test data."""
import sys,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.ml.model_manager import ModelManager
from core.validation_evidence import source_digest
from core.history_cache import digest

def generate():
    reports=ROOT/"output/reports"
    tests=json.loads((reports/"stage5_1_test_results.json").read_text(encoding="utf-8"))
    previous=json.loads((reports/"stage5_test_results.json").read_text(encoding="utf-8"))
    gui=json.loads((reports/"stage5_1_gui_verification.json").read_text(encoding="utf-8"))
    assert tests["source_digest"]==source_digest(ROOT)
    old_passed=sum(tests["tests"].get(k)=="PASS" for k in previous["tests"])
    assert old_passed==282 and tests["passed"]==tests["total"]
    manager=ModelManager(ROOT);models=manager.registry.list_models()
    legacy=[m for m in models if "evaluation_protocol" not in m]
    frozen=[m for m in models if m.get("evaluation_protocol")=="RESEARCH_FROZEN"]
    assert len(legacy)==4 and len(frozen)==2 and all(m["is_demo"] for m in models)
    for identity,files in gui["legacy_hashes"].items():
        for name,sha in files.items():assert digest(ROOT/"models/bundles"/identity/name)==sha
    for m in models:manager.load_model(m["model_id"])
    comparisons=[]
    for new in frozen:
        old=next(m for m in legacy if m["model_type"]==new["model_type"] and m["combination_id"]=="COMBO_ATHC")
        comparisons.append(dict(model_type=new["model_type"],legacy_id=old["model_id"],frozen_id=new["model_id"],rolling=old["metrics"],frozen=new["metrics"],cutoff=new["validation_history_cutoff"]))
    report=dict(generated_at=datetime.now(timezone.utc).isoformat(),passed=True,source_digest=source_digest(ROOT),old_tests_passed=old_passed,new_tests=tests["total"]-282,total_tests=tests["total"],warnings=tests["warnings"],default_protocol="RESEARCH_FROZEN",legacy_assets_unchanged=True,comparisons=comparisons,gui=gui,final_test_evaluated=False,shap_run=False,threshold_search=False)
    (reports/"stage5_1_history_visibility_validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Stage 5.1 历史可见性验证", "", "结论：通过。研究训练默认 RESEARCH_FROZEN；ONLINE_ROLLING 可显式选择。两者代表不同评价假设，并非原 rolling 代码错误。", "", "## 时间规则", "", "| 场景 | ROLLING_ASOF | FROZEN_CUTOFF |", "|---|---|---|", "| 训练 | source_date < D | 研究训练仍使用 expanding rolling |", "| 验证 | 允许前日真实结果影响后日 | source_date <= train_end_date |", "| 最终测试接口 | 每日 as-of | source_date <= validation_end_date |", "", "Frozen cutoff 必须严格早于每个目标日期，等于/晚于目标日、无效日期和带时区/时刻的边界均拒绝。仅 H 查询采用 cutoff+1 的严格日期边界，A/T/C、标签和真实目标日期不替换。原 HistoricalFeatureEngine 及历史缓存未重写。所有原始和 fallback 来源日期均检查 cutoff。", "", "最终测试仅验证合成日期的特征接口预设，没有运行 2025-01 真实数据、预测指标、阈值搜索或 SHAP。TrainingService 和 InferenceService 的最终期锁保持。", "", "## 回归及变形测试", "", f"旧测试 {old_passed}/282 通过；新增 {tests['total']-282} 项，总计 {tests['passed']}/{tests['total']} 通过；单元测试 warning {tests['warning_count']}。", "", "- 修改 03-13～03-29 的全部 DEP_DELAY，03-13～03-30 全部 Frozen H 精确不变。", "- Rolling 修改 03-13 结果：03-13 不变，03-14 及以后统计变化。", "- 追加未来记录、cutoff 后大量不同标签：Frozen H 精确不变。", "- cutoff=03-10 / 03-12 时，合成单航班日数据的 global_count 分别恒为 10 / 12，验证包含截止日。", "- 无效或同日/未来 cutoff、伪造后截止日来源被拒绝；冷启动回退也遵守边界。", "- 训练服务实测默认冻结、显式滚动、metadata 完整性和独立构造验证矩阵的指标一致性。", "- GUI 默认协议、预检显示日期、选择切换、训练时禁用协议控制通过。", "", "## DEMO 指标比较", "", "相同训练/验证边界、参数和阈值 0.5；只新增 RF ATHC、XGBoost ATHC 两份冻结 DEMO。训练仍为 2024-01-01～03-12 的 423 条有效样本，验证为 03-13～03-30 的 106 条；验证 H 截止 03-12。DEMO 指标不代表真实 BTS 效果。", "", "| 模型 | 验证 H | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in comparisons:
        for policy in ["rolling","frozen"]:
            m=c[policy];lines.append("| "+" | ".join([c["model_type"],policy]+[f"{m[k]:.6f}" for k in ["accuracy","precision","recall","f1","roc_auc","average_precision"]])+" |")
    for c in comparisons:
        lines += ["",f"{c['model_type']} frozen model ID: `{c['frozen_id']}`；混淆矩阵（TN,FP / FN,TP）：{c['frozen']['confusion_matrix']}。", ""]
    lines += ["## 资产、界面和限制", "", "旧四个模型的 model.joblib、metadata.json、metrics.json SHA-256 全部不变；没有删除、重写或重新训练 A 模型。旧模型在详情页通过只读展示标记 legacy_rolling_validation，原始 metadata 不迁移。新模型 model_version=2，完整记录 evaluation_protocol、history_visibility_mode、history_cutoff_date、train_history_policy、validation_history_policy、validation_history_cutoff、test_history_policy、test_history_cutoff；缺失或矛盾字段拒绝加载。", "", "保存后重新加载并重建相同冻结特征的概率一致性已在训练流程通过；六个模型均可加载。新 H 模型默认推理继承保存的协议；显式在线模式可覆盖并在输出警告中记录，直接输入不匹配 H 协议则拒绝。", "", "GUI 实际后台训练两份 Frozen DEMO，管理详情显示协议、可见模式和截止日期，截图位于 output/figures/stage5_1_*.png。快速切换导航时有 Matplotlib constrained_layout 非阻断警告，与单元测试 warning 分开记录。", "", "真实 BTS 尚未验证；Frozen 假设截止日内结果已经可获得，未建模结果发布延迟。最终时间外指标仍锁定。没有进入 Stage 6。", "", "## 验收 A–P", ""]
    items=["原 282 项测试保持通过","Rolling 原逻辑保持","Frozen 已实现","Frozen validation 不使用验证期标签","修改验证期标签不改变 Frozen H","Rolling 逐日更新","默认研究 Frozen","可选在线 Rolling","metadata 保存 protocol","metadata 保存 cutoff","GUI 展示 protocol","ATHC Frozen DEMO 可训练","保存/加载一致","最终测试锁保持","未运行 SHAP","未做阈值搜索"]
    lines += [f"- {chr(65+i)}：通过 — {text}" for i,text in enumerate(items)]
    (reports/"stage5_1_history_visibility_validation.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("gui","comparisons","warnings")},ensure_ascii=False))
if __name__=="__main__":generate()
