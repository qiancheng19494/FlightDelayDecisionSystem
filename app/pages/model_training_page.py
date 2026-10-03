"""后台训练与可信模型管理；所有指标来自训练服务。"""
import json
import pandas as pd
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QPushButton,QCheckBox,QPlainTextEdit,QProgressBar,QMessageBox
from app.widgets.data_table import DataTable
from app.workers.task_worker import TaskController
from core.feature_groups import FeatureCombinationRegistry
from core.historical_features import json_safe
from core.history_normalization import normalize_frame
from core.data_loader import FlightDataLoader
from core.ml.time_split import TemporalDataSplitter
from core.ml.training_service import train_dataset,load_model_config
from core.ml.model_manager import ModelManager
from core.ml.trainers import xgboost_status
from core.history_visibility import EvaluationProtocol,history_description
DEMO_NOTICE="DEMO 模型，仅用于软件功能验证，其性能不代表真实 BTS 航班数据表现。"
class ModelTrainingPage(QWidget):
    def __init__(self,context,on_saved):
        super().__init__();self.context=context;self.on_saved=on_saved;self.result=None;self.task=TaskController(self)
        layout=QVBoxLayout(self);layout.setSpacing(10)
        self.notice=QLabel(DEMO_NOTICE);self.notice.setWordWrap(True);self.notice.setStyleSheet("background:#fff1cc;color:#7b4100;padding:12px;font-weight:bold;");layout.addWidget(self.notice)
        row=QHBoxLayout();self.dataset=QComboBox();self.history=QComboBox();self.schedule=QComboBox()
        for name,box in [("训练数据",self.dataset),("历史缓存",self.history),("已知计划表",self.schedule)]:
            box.setMinimumContentsLength(12);box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon);row.addWidget(QLabel(name));row.addWidget(box,1)
        layout.addLayout(row);row=QHBoxLayout();self.combo=QComboBox()
        for c in FeatureCombinationRegistry().list_combinations():self.combo.addItem(c.name,c.combination_id)
        self.combo.setCurrentIndex(self.combo.findData("COMBO_ATHC"));self.model=QComboBox();self.model.addItem("Random Forest","rf")
        ok,reason=xgboost_status();self.model.addItem("XGBoost" if ok else "XGBoost · 依赖不可用","xgboost")
        if not ok:self.model.model().item(1).setEnabled(False)
        self.estimate_button=QPushButton("预检时间划分");self.train_button=QPushButton("开始训练")
        for control in [self.combo,self.model,self.estimate_button,self.train_button]:row.addWidget(control)
        layout.addLayout(row);self.formal=QCheckBox("明确启动所选非 DEMO 数据的正式训练（2024-01～11 / 2024-12）");layout.addWidget(self.formal)
        protocol_row=QHBoxLayout();protocol_row.addWidget(QLabel("评价协议"));self.protocol=QComboBox()
        self.protocol.addItem("研究冻结模式",EvaluationProtocol.RESEARCH_FROZEN.value);self.protocol.addItem("在线滚动模式",EvaluationProtocol.ONLINE_ROLLING.value)
        protocol_row.addWidget(self.protocol);layout.addLayout(protocol_row)
        self.split_metadata=None;self.policy_label=QLabel();self.policy_label.setWordWrap(True);layout.addWidget(self.policy_label)
        self.protocol.currentIndexChanged.connect(self.update_policy);self.update_policy()
        self.range_label=QLabel();self.range_label.setWordWrap(True);layout.addWidget(self.range_label)
        self.status=QLabel("固定 threshold=0.50；2025-01 及之后锁定；无 SHAP 或阈值搜索。");self.status.setWordWrap(True);layout.addWidget(self.status)
        self.progress=QProgressBar();layout.addWidget(self.progress)
        self.summary=QLabel("尚未训练模型。");self.summary.setWordWrap(True);layout.addWidget(self.summary)
        self.metrics=DataTable();self.metrics.setMinimumHeight(260);self.metrics.setMaximumHeight(295);layout.addWidget(self.metrics)
        self.details=QPlainTextEdit();self.details.setReadOnly(True);self.details.setMaximumHeight(135);layout.addWidget(self.details)
        self.task.status.connect(self.status.setText);self.task.progress.connect(lambda p,n:self.progress.setValue(p));self.task.busy_changed.connect(self.busy);self.task.failed.connect(self.failed)
        self.dataset.currentIndexChanged.connect(self.selection);self.train_button.clicked.connect(self.train);self.estimate_button.clicked.connect(self.estimate)
        self.refresh()
    def update_policy(self):
        frozen=self.protocol.currentData()=="RESEARCH_FROZEN"
        end=self.split_metadata["training_end_date"] if self.split_metadata else "待时间划分预检"
        self.policy_label.setText(f"训练结束日期：{end}；验证历史截止日期：{end}。验证期历史特征不会吸收验证期内已发生的真实标签。" if frozen else "在线滚动模式：H 使用目标日之前已观察到的历史，允许前一验证日结果影响后日。")
    def estimated(self,m):
        self.split_metadata=m;self.update_policy()
        self.range_label.setText(f"训练 {m['training_start_date']}～{m['training_end_date']}：{m['train_rows_before_label_filter']} 行；验证 {m['validation_start_date']}～{m['validation_end_date']}：{m['validation_rows_before_label_filter']} 行（剔除无效标签前）。")
    def refresh(self):
        if self.task.busy:return
        self.records=self.context.datasets.datasets()
        for box in [self.dataset,self.history,self.schedule]:
            old=box.currentData();box.blockSignals(True);box.clear()
            if box is not self.dataset:box.addItem("未选择",None)
            for r in self.records:
                if box is not self.schedule and r["dataset_type"]!="HISTORICAL_LABELED":continue
                box.addItem(f"{r['id']} · {r['dataset_name']}",r["id"])
            box.setCurrentIndex(max(0,box.findData(old)));box.blockSignals(False)
        self.selection()
    def selection(self):
        self.split_metadata=None;self.update_policy()
        r=next((r for r in self.records if r["id"]==self.dataset.currentData()),None)
        if r:self.range_label.setText(f"源范围 {r['start_date']} 至 {r['end_date']} · {r['row_count']} 行 · 有效标签 {r['valid_labeled_count']}；点击预检查看实际唯一日期分界与分区行数。")
        else:self.range_label.setText("请先导入有标签数据。")
        self.notice.setText(DEMO_NOTICE if r is None or r["is_demo"] else "非 DEMO 数据；仅接受用户明确启动，沿用论文日期窗口；不代表已验证论文模型。")
        self.busy(False)
    def busy(self,b):
        for control in [self.dataset,self.history,self.schedule,self.combo,self.model,self.protocol,self.formal,self.estimate_button]:control.setEnabled(not b)
        self.train_button.setEnabled(not b and self.dataset.currentData() is not None)
    def failed(self,text):self.status.setText("训练/预检未完成："+text);QMessageBox.warning(self,"模型任务未完成",text)
    def estimate(self):
        identity=self.dataset.currentData()
        if identity is None:return
        def work(progress,cancel):
            r=self.context.datasets.repository.get(identity);parts=[];n=0
            for chunk in FlightDataLoader(r["file_path"],10000).iter_chunks(cancelled=cancel):
                n+=len(chunk)
                if n>load_model_config(self.context.root)["max_training_rows"]:raise ValueError("超过本阶段训练上限")
                parts.append(normalize_frame(chunk).flight_date)
            if not parts:raise ValueError("数据为空")
            return TemporalDataSplitter().split(pd.concat(parts,ignore_index=True),is_demo=bool(r["is_demo"])).metadata
        self.task.start(work,"检查唯一日期边界",self.estimated)
    def train(self):
        args=(self.context,self.dataset.currentData(),self.history.currentData(),self.schedule.currentData(),self.combo.currentData(),self.model.currentData());formal=self.formal.isChecked();protocol=self.protocol.currentData()
        self.progress.setValue(0)
        self.task.start(lambda progress,cancel,stage:train_dataset(*args,formal_requested=formal,evaluation_protocol=protocol,progress=progress,cancelled=cancel,stage=stage),"准备训练",self.ready,with_stage=True)
    def ready(self,result):
        self.result=result;self.show_metadata(result["metadata"],result["model_path"],result["total_seconds"]);self.progress.setValue(100);self.status.setText("训练、保存、加载与内部推理校验完成；未自动激活。");self.on_saved()
    def show_metadata(self,m,path="",elapsed=None):
        self.policy_label.setText(history_description(m))
        self.progress.setValue(100);self.status.setText("显示已保存的模型结果；指标来自 metadata，不进行阈值搜索。")
        self.notice.setText(DEMO_NOTICE if m["is_demo"] else "非 DEMO 训练模型；尚未进行最终时间外评价。")
        self.summary.setText(f"{m['model_name']}\nID：{m['model_id']}\n训练 {m['training_row_count']} 行（正类 {m['positive_rate_train']:.2%}），验证 {m['validation_row_count']} 行（正类 {m['positive_rate_validation']:.2%}） · threshold=0.50")
        self.range_label.setText(f"训练 {m['training_start_date']}～{m['training_end_date']}；验证 {m['validation_start_date']}～{m['validation_end_date']}")
        metrics=m["metrics"];self.metrics.set_frame(pd.DataFrame([{"DEMO VALIDATION METRICS" if m["is_demo"] else "VALIDATION METRICS":key,"实际值":"N/A" if metrics[key] is None else (str(metrics[key]) if key=="confusion_matrix" else f"{metrics[key]:.6f}")} for key in ["accuracy","precision","recall","f1","roc_auc","average_precision","confusion_matrix"]]))
        self.metrics.setColumnWidth(0,355);self.metrics.setColumnWidth(1,430)
        self.details.setPlainText(f"模型路径：{path}\n耗时：{elapsed if elapsed is not None else '见 metadata.timings'} 秒\n"+json.dumps(json_safe({"timings":m["timings"],"warnings":m["warnings"],"class_weight":m["adopted_weight"]}),ensure_ascii=False,indent=2))
class ModelManagementPage(QWidget):
    def __init__(self,context):
        super().__init__();self.context=context;self.manager=ModelManager(context.root);self.task=TaskController(self);self.loaded=None
        layout=QVBoxLayout(self);self.notice=QLabel("仅加载本项目生成的可信 joblib/pickle；不自动加载网络或未验证外部模型。");self.notice.setWordWrap(True);layout.addWidget(self.notice)
        row=QHBoxLayout();self.selector=QComboBox();self.selector.setMinimumContentsLength(20);self.selector.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon);row.addWidget(self.selector,1)
        self.refresh_button=QPushButton("刷新");self.load_button=QPushButton("加载");self.active_button=QPushButton("设为当前模型");self.meta_button=QPushButton("查看 metadata");self.metrics_button=QPushButton("查看 metrics");self.archive_button=QPushButton("归档");self.protocol_button=QPushButton("论文协议审计")
        self.buttons=[self.refresh_button,self.load_button,self.active_button,self.meta_button,self.metrics_button,self.archive_button,self.protocol_button]
        for b in self.buttons:row.addWidget(b)
        layout.addLayout(row);self.status=QLabel("未加载模型");layout.addWidget(self.status)
        self.protocol_audit_view=QPlainTextEdit();self.protocol_audit_view.setReadOnly(True);self.protocol_audit_view.setMaximumHeight(130);layout.addWidget(self.protocol_audit_view)
        self.table=DataTable();self.table.setFixedHeight(195);layout.addWidget(self.table)
        self.details=QPlainTextEdit();self.details.setReadOnly(True);self.details.setMinimumHeight(320);layout.addWidget(self.details)
        self.protocol_button.clicked.connect(self.show_protocol_audit)
        self.refresh_button.clicked.connect(self.refresh);self.load_button.clicked.connect(lambda:self.action("load"));self.active_button.clicked.connect(lambda:self.action("active"));self.archive_button.clicked.connect(lambda:self.action("archive"));self.meta_button.clicked.connect(lambda:self.show("metadata"));self.metrics_button.clicked.connect(lambda:self.show("metrics"));self.selector.currentIndexChanged.connect(lambda:self.show("metadata"))
        self.task.busy_changed.connect(lambda b:[x.setEnabled(not b) for x in [self.selector,*self.buttons]]);self.task.failed.connect(lambda text:self.status.setText("拒绝操作："+text));self.refresh()
    def refresh(self):
        if self.task.busy:return
        records=self.manager.registry.list_models();old=self.selector.currentData();self.selector.blockSignals(True);self.selector.clear()
        for m in records:self.selector.addItem(m["model_name"]+" · "+m["model_id"][-8:],m["model_id"])
        self.selector.setCurrentIndex(max(0,self.selector.findData(old)));self.selector.blockSignals(False)
        active=self.manager.registry.read()["active_model_id"]
        self.table.set_frame(pd.DataFrame([{"模型 ID":m["model_id"],"类型":m["model_type"],"组合":m["combination_id"],"状态":m["status"],"性质":"DEMO MODEL" if m["is_demo"] else m["status"],"验证 AP":round(m["metrics"]["average_precision"],6) if m["metrics"]["average_precision"] is not None else "N/A","验证 AUC":round(m["metrics"]["roc_auc"],6) if m["metrics"]["roc_auc"] is not None else "N/A","当前":m["model_id"]==active,"训练期":m["training_start_date"]+"～"+m["training_end_date"],"验证期":m["validation_start_date"]+"～"+m["validation_end_date"],"创建":m["created_at"]} for m in records]));
        for i,width in enumerate([335,75,120,75,110,100,100,60,210,210,210]):self.table.setColumnWidth(i,width)
        self.show("metadata")
    def show(self,kind):
        identity=self.selector.currentData()
        if identity is None:self.details.setPlainText("暂无模型资产。外部未验证资产不会自动加载。");return
        m=self.manager.registry.get_model(identity)
        from pathlib import Path
        report=Path(self.context.root)/"output/reports/stage5_2_protocol_audit.json"
        try:paper=json.loads(report.read_text(encoding="utf-8"))
        except (OSError,ValueError):paper={}
        self.protocol_audit_view.setPlainText("评价协议：RESEARCH_FROZEN（验证冻结） / ONLINE_ROLLING（验证逐日更新）\n"+history_description(m)+"\n论文历史实验 protocol："+paper.get("protocol","未确认")+" · "+paper.get("confidence","UNRESOLVED")+"\n论文验证："+paper.get("validation_protocol","未确认")+"；论文最终期："+paper.get("test_history","未确认")+"\n论文 16 实验为 8 组合×2模型；详情与证据见 Stage 5.2 protocol audit 报告。" if paper.get("combinations",{}).get("status")=="CONFIRMED" else "评价协议：RESEARCH_FROZEN / ONLINE_ROLLING\n"+history_description(m)+"\n论文历史实验 protocol：未确认")
        self.notice.setText(DEMO_NOTICE+" 仅加载可信项目资产。" if m["is_demo"] else (m.get("model_identity_notice") or "仅加载可信项目资产；不得对不可信 joblib/pickle 反序列化。"))
        self.details.setPlainText((history_description(m)+"\n\n" if kind=="metadata" else "")+json.dumps(m if kind=="metadata" else m["metrics"],ensure_ascii=False,indent=2))
    def show_protocol_audit(self):
        from pathlib import Path
        report=Path(self.context.root)/"output/reports/stage5_2_protocol_audit.json"
        try:
            p=json.loads(report.read_text(encoding="utf-8"))
            lines=["论文评价协议审计（只读；未重算或修改论文）",f"整体：{p['protocol']} · 证据等级：{p['confidence']}",f"验证：{p['validation_protocol']}",f"最终期：{p['test_history']}"]
            for item in p.get("cache_date_checks",[]):lines.append(f"目标 {item['target_date']} → 最晚来源 {item['max_source_date']}（检查 {item['rows']} 条）")
            lines += [p.get("paper_text_assessment","无法判断"),"组合："+str(p.get("combinations",{})),"证据来源："]+[e["path"] for e in p.get("evidence",[])]
            self.details.setPlainText("\n".join(lines))
        except (OSError,ValueError,KeyError):self.details.setPlainText("论文历史实验 protocol：未确认；尚无有效审计报告。")
    def action(self,kind):
        identity=self.selector.currentData()
        if identity is None:return
        def work(progress,cancel):
            if kind=="load":return self.manager.load_model(identity)
            if kind=="active":self.manager.registry.set_active_model(identity)
            else:self.manager.registry.archive_model(identity)
            return None
        def done(bundle):
            self.loaded=bundle;self.status.setText({"load":"模型已加载（已检查 schema / 指纹 / 预处理器）","active":"已设置当前模型","archive":"已归档；文件保留"}[kind]);self.refresh()
        self.task.start(work,"正在检查模型资产",done)
