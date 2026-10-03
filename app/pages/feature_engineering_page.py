"""模型性能页内的特征开发区，保持 15 项导航，复用统一 Pipeline。"""
import json
import pandas as pd
from PySide6.QtWidgets import QHBoxLayout,QComboBox,QPushButton,QLabel,QTabWidget,QWidget,QVBoxLayout,QPlainTextEdit,QFileDialog,QMessageBox,QProgressBar
from app.pages.base_page import BasePage
from app.widgets.data_table import DataTable
from app.workers.task_worker import TaskController
from core.feature_groups import FeatureCombinationRegistry,FEATURE_DEFINITIONS,GROUP_FIELDS
from core.feature_service import build_feature_sample,feature_profile,audit_result,export_feature_sample
class FeatureEngineeringPage(BasePage):
    def __init__(self,context):
        super().__init__("特征工程 · 开发验证","本区域仅生成和验证模型输入特征，不进行模型训练或效果评价。")
        self.context=context;self.result=None;self.body.setSpacing(10)
        self.task=TaskController(self);self.task.busy_changed.connect(self.set_busy);self.task.status.connect(lambda text:self.status.setText(text));self.task.failed.connect(self.failed)
        self.task.progress.connect(lambda percent,rows:self.progress.setValue(percent))
        self.selectors=QHBoxLayout();self.dataset=QComboBox();self.history=QComboBox();self.schedule=QComboBox()
        for label,box in [("目标数据",self.dataset),("历史缓存",self.history),("已知计划表",self.schedule)]:
            box.setMinimumContentsLength(10);box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon);self.selectors.addWidget(QLabel(label));self.selectors.addWidget(box,1)
        self.body.addLayout(self.selectors)
        row=QHBoxLayout();self.combination=QComboBox()
        for c in FeatureCombinationRegistry().list_combinations():self.combination.addItem(c.name,c.combination_id)
        self.combination.setCurrentIndex(self.combination.findData("COMBO_ATHC"));row.addWidget(self.combination,1)
        self.refresh_button=QPushButton("刷新数据集");self.build_button=QPushButton("生成特征预览");self.export_button=QPushButton("导出特征样本");self.audit_button=QPushButton("运行特征检查")
        for b in [self.refresh_button,self.build_button,self.export_button,self.audit_button]:row.addWidget(b)
        self.body.addLayout(row);self.counts=QLabel();self.body.addWidget(self.counts)
        self.notice=QLabel("C 是目标日已知完整计划表的代理，不是实时交通量；选择计划表即确认其在计划阶段已知。未选择时 C 为 NaN。H 仅使用 date < D。")
        self.notice.setWordWrap(True);self.body.addWidget(self.notice)
        self.status=QLabel("先选择数据与已有历史缓存。仅预览前 100 行；不自动创建历史缓存。");self.status.setWordWrap(True);self.body.addWidget(self.status)
        self.progress=QProgressBar();self.progress.setValue(0);self.body.addWidget(self.progress)
        self.tabs=QTabWidget();self.tabs.setFixedHeight(440);self.body.addWidget(self.tabs)
        self.overview=QPlainTextEdit();self.overview.setReadOnly(True);self.tabs.addTab(self.overview,"概览")
        self.groups=DataTable();self.tabs.addTab(self.groups,"字段定义")
        self.preview=DataTable();self.tabs.addTab(self.preview,"特征预览")
        self.registry=DataTable();self.tabs.addTab(self.registry,"16 组合")
        self.audit=DataTable();self.tabs.addTab(self.audit,"特征检查")
        self.congestion=DataTable();self.tabs.addTab(self.congestion,"计划拥堵示例")
        self.registry.set_frame(pd.DataFrame([{"组合":c.name,"ID":c.combination_id,"字段数":len(c.feature_names),"论文定义":"通用子集，非论文实验表","用途":"特征配置；未训练模型"} for c in FeatureCombinationRegistry().list_combinations()]))
        self.registry.verticalHeader().setMinimumSectionSize(20);self.registry.verticalHeader().setDefaultSectionSize(22)
        for i,width in enumerate([225,175,95,120,260]):self.registry.setColumnWidth(i,width)
        self.build_button.clicked.connect(self.build);self.refresh_button.clicked.connect(self.refresh);self.export_button.clicked.connect(self.export);self.audit_button.clicked.connect(self.run_audit)
        for box in [self.dataset,self.history,self.schedule,self.combination]:box.currentIndexChanged.connect(self.invalidate)
        self.refresh();self.invalidate();self.body.addStretch()
    def refresh(self):
        if self.task.busy:return
        records=self.context.datasets.datasets()
        for box in [self.dataset,self.history,self.schedule]:
            old=box.currentData();box.blockSignals(True);box.clear()
            if box is not self.dataset:box.addItem("未选择",None)
            for r in records:
                if box is self.history and r["dataset_type"]!="HISTORICAL_LABELED":continue
                box.addItem(f"{r['id']} · {r['dataset_name']}",r["id"])
            idx=box.findData(old);box.setCurrentIndex(max(0,idx));box.blockSignals(False)
        self.invalidate()
    def invalidate(self):
        self.result=None
        c=FeatureCombinationRegistry().get_combination(self.combination.currentData())
        self.counts.setText("  ·  ".join(f"{g}: {len(GROUP_FIELDS[g]) if g in c.groups else 0} 字段" for g in GROUP_FIELDS)+f"  ·  合计 {len(c.feature_names)}  · schema 1")
        self.overview.setPlainText("选择配置后生成真实特征。软件通用 16 子集框架；研究协议记录见模型管理。")
        for table in [self.groups,self.preview,self.audit,self.congestion]:table.set_frame(pd.DataFrame())
        self.set_busy(False)
    def set_busy(self,busy):
        for box in [self.dataset,self.history,self.schedule,self.combination]:box.setEnabled(not busy)
        self.build_button.setEnabled(not busy and self.dataset.currentData() is not None);self.refresh_button.setEnabled(not busy)
        for button in [self.audit_button,self.export_button]:button.setEnabled(not busy and self.result is not None)
    def failed(self,text):self.status.setText("未完成："+text);QMessageBox.warning(self,"特征任务未完成",text)
    def build(self):
        identity=self.dataset.currentData();history=self.history.currentData();schedule=self.schedule.currentData();combo=self.combination.currentData()
        self.progress.setValue(0)
        self.task.start(lambda progress,cancel:build_feature_sample(self.context,identity,history,combo,schedule,100,progress,cancel),"正在生成样本；计划表按块聚合",self.ready)
    def ready(self,result):
        self.result=result;self.preview.set_frame(result.X.head(100));self.preview.resizeColumnsToContents();self.groups.set_frame(feature_profile(result))
        for i,width in enumerate([310,75,95,85,130,350]):self.groups.setColumnWidth(i,width)
        counts=result.metadata["group_feature_counts"]
        self.overview.setPlainText(f"数据集 {result.metadata['dataset_id']}：{result.metadata['dataset_name']}\n组合：{result.metadata['combination_id']}\n实际生成：{len(result.X)} 行 × {len(result.X.columns)} 字段\nA/T/H/C：{counts}\n历史缓存：{result.metadata['historical_cache_id']}\n完整计划上下文扫描：{result.metadata['schedule_context_rows']} 行\nX/y 独立，缺失不填全局均值，不进行模型训练。\n研究协议记录见模型管理。\n\n提示："+"\n".join(result.warnings))
        cfields=[n for n in result.X if FEATURE_DEFINITIONS[n].group=="C"]
        display=result.X[[n for n in ("airline","origin","destination") if n in result.X]+cfields].head(100).copy()
        display=display.rename(columns={"airline":"航司","origin":"始发","destination":"目的","origin_hour_scheduled_departures":"始发小时量","destination_hour_scheduled_arrivals":"目的小时量","airline_hour_scheduled_departures":"航司小时量","route_day_scheduled_flights":"有向航线日量","origin_day_scheduled_departures":"始发日量","destination_day_scheduled_arrivals":"目的日量","congestion_context_available":"上下文可用"})
        if result.history_metadata is not None:display.insert(0,"目标日期",result.history_metadata.target_date.head(100).dt.strftime("%Y-%m-%d"))
        self.congestion.set_frame(display)
        for i in range(len(display.columns)):self.congestion.setColumnWidth(i,110 if i==0 else 100)
        self.status.setText("生成完成；类别字段保留原值，NaN 保留。可检查或导出 X/y 分离样本。");self.progress.setValue(100);self.set_busy(False)
    def run_audit(self):
        if self.result:
            r=audit_result(self.result);self.audit.set_frame(pd.DataFrame([{"检查":i["name"],"结果":"PASS" if i["passed"] else "FAIL"} for i in r["checks"]]));self.audit.setColumnWidth(0,510)
            self.status.setText("特征检查通过" if r["passed"] else "特征检查失败");self.tabs.setCurrentIndex(4)
    def export(self):
        if self.result is None:return
        path=QFileDialog.getExistingDirectory(self,"选择样本导出目录",str(self.context.root/"output/features"))
        if path:self.task.start(lambda progress,cancel:export_feature_sample(self.result,path),"正在导出独立 X/y 与 metadata",lambda folder:self.status.setText("已导出："+str(folder)))
