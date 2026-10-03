"""历史统计 GUI；计算和审计使用独立引擎及现有线程任务。"""
import json
import time
from pathlib import Path
import pandas as pd
from PySide6.QtCore import QDate, QTimer
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QComboBox,QPushButton,QLabel,QProgressBar,QTabWidget,QPlainTextEdit,QDateEdit,QLineEdit,QSpinBox,QFileDialog,QMessageBox)
from app.pages.base_page import BasePage
from app.pages.data_manager_page import percent
from app.widgets.metric_card import MetricCard
from app.widgets.data_table import DataTable
from app.widgets.history_chart_widget import HistoryChartWidget
from app.workers.task_worker import TaskController
from core.historical_features import HistoricalFeatureEngine,json_safe
from core.history_schema import ENTITIES
from core.history_report import export_history_report

ENTITY_NAMES={"global":"全局","airline":"航司","origin":"始发机场","destination":"目的机场","route":"航线","airline_origin":"航司-始发机场"}

class HistoryAnalysisPage(BasePage):
    def __init__(self,context):
        super().__init__("历史运行统计","日级严格历史窗口：所有目标日期 D 的特征仅使用 date < D 的记录。")
        self.body.setSpacing(12)
        self.context=context
        self.engine=None
        self.records=[]
        self.entity_rows=pd.DataFrame()
        self.query_result=None
        self.started=0
        self.stage_text="就绪";self.scanned=0
        self.task=TaskController(self)
        self.task.progress.connect(self.on_progress)
        self.task.status.connect(self.set_stage)
        self.task.failed.connect(self.on_error)
        self.task.cancelled.connect(lambda:self.set_stage("任务已取消；旧缓存保留"))
        self.task.busy_changed.connect(self.set_busy)
        self.timer=QTimer(self);self.timer.setInterval(250);self.timer.timeout.connect(self.update_status)
        top=QHBoxLayout()
        self.selector=QComboBox();self.selector.setMinimumContentsLength(28);self.selector.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.build_button=QPushButton("构建历史统计")
        self.refresh_button=QPushButton("刷新")
        self.report_button=QPushButton("生成验证报告")
        self.rebuild_button=QPushButton("重新构建")
        self.audit_button=QPushButton("运行泄漏检查")
        self.cancel_button=QPushButton("取消")
        top.addWidget(self.selector,1)
        for button in [self.build_button,self.refresh_button,self.rebuild_button,self.audit_button,self.report_button,self.cancel_button]:top.addWidget(button)
        self.body.addLayout(top)
        self.status=QLabel("请选择一个历史数据集；不会自动跨数据集合并。")
        self.status.setWordWrap(True);self.body.addWidget(self.status)
        self.progress=QProgressBar();self.progress.setValue(0);self.body.addWidget(self.progress)
        self.summary=QLabel("尚未构建历史缓存");self.summary.setWordWrap(True);self.body.addWidget(self.summary)
        grid=QGridLayout();self.cards=[]
        for i,title in enumerate(["源记录数","有效历史标签","日期范围","航司数","始发 / 目的机场","有向航线数"]):
            card=MetricCard(title,"—");card.setMaximumHeight(90);card.value_label.setObjectName("compactMetricValue");card.value_label.setWordWrap(True)
            grid.addWidget(card,i//3,i%3);self.cards.append(card)
        self.body.addLayout(grid)
        self.tabs=QTabWidget();self.body.addWidget(self.tabs)
        self.charts=HistoryChartWidget();self.tabs.addTab(self.charts,"概览趋势（描述性）")
        self._entity_tab();self._query_tab();self._audit_tab()
        self.tabs.setFixedHeight(340)
        disclaimer=QLabel("历史延误率和历史平均延误为过去运行数据的描述性统计，不代表因果关系，也不构成航空安全评价。")
        disclaimer.setWordWrap(True);disclaimer.setObjectName("muted");self.body.addWidget(disclaimer)
        self.build_button.clicked.connect(lambda:self.build(False))
        self.rebuild_button.clicked.connect(lambda:self.build(True))
        self.refresh_button.clicked.connect(self.load_cache)
        self.audit_button.clicked.connect(self.run_audit)
        self.report_button.clicked.connect(self.export_report)
        self.cancel_button.clicked.connect(self.task.cancel)
        self.selector.currentIndexChanged.connect(self.selection_changed)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.refresh()

    def tab_changed(self,index):
        self.summary.setVisible(index==0)
        for card in self.cards:card.setVisible(index==0)
        self.tabs.setFixedHeight(340 if index==0 else 500)
        if index==1:self.load_entities()

    def _entity_tab(self):
        widget=QWidget();layout=QVBoxLayout(widget)
        operations=QHBoxLayout()
        self.entity_selector=QComboBox()
        for key in ["airline","origin","destination","route","airline_origin"]:self.entity_selector.addItem(ENTITY_NAMES[key],key)
        self.sort_selector=QComboBox();self.sort_selector.addItem("按航班量","count");self.sort_selector.addItem("按延误率","delay_rate")
        self.minimum=QSpinBox();self.minimum.setRange(1,10000000);self.minimum.setPrefix("展示最少样本：")
        self.entity_button=QPushButton("更新列表")
        for control in [self.entity_selector,self.sort_selector,self.minimum,self.entity_button]:operations.addWidget(control)
        layout.addLayout(operations)
        self.entity_label=QLabel("覆盖整个选定数据集；展示排序前 500 个实体，不改变底层统计。")
        self.entity_label.setWordWrap(True);layout.addWidget(self.entity_label)
        self.entity_table=DataTable();self.entity_table.setFixedHeight(140);layout.addWidget(self.entity_table)
        self.entity_chart=HistoryChartWidget();self.entity_chart.setMinimumHeight(210);layout.addWidget(self.entity_chart)
        self.tabs.addTab(widget,"实体统计")
        self.entity_button.clicked.connect(self.load_entities)

    def _query_tab(self):
        widget=QWidget();layout=QVBoxLayout(widget);controls=QHBoxLayout()
        self.query_date=QDateEdit(QDate(2024,1,3));self.query_date.setCalendarPopup(True);self.query_date.setDisplayFormat("yyyy-MM-dd")
        self.query_airline=QLineEdit("AA");self.query_origin=QLineEdit("AAA");self.query_destination=QLineEdit("BBB")
        for label,control in [("日期",self.query_date),("航司",self.query_airline),("始发",self.query_origin),("目的",self.query_destination)]:
            controls.addWidget(QLabel(label));controls.addWidget(control)
            if isinstance(control,QLineEdit): control.setMaximumWidth(90)
        self.query_button=QPushButton("查询历史特征");controls.addWidget(self.query_button)
        layout.addLayout(controls)
        self.query_notice=QLabel("所有历史统计仅使用目标日期之前的航班数据（date < target date），不使用目标日期当天或未来标签。样本数为原实体，率和均值为回退后有效值。")
        self.query_notice.setWordWrap(True);layout.addWidget(self.query_notice)
        self.query_table=DataTable();self.query_table.setMinimumHeight(220);layout.addWidget(self.query_table)
        self.query_details=QPlainTextEdit();self.query_details.setReadOnly(True);self.query_details.setMaximumHeight(100);layout.addWidget(self.query_details)
        self.tabs.addTab(widget,"历史特征查询")
        self.query_button.clicked.connect(self.query_history)

    def _audit_tab(self):
        widget=QWidget();layout=QVBoxLayout(widget)
        self.audit_label=QLabel("尚未运行审计");self.audit_label.setWordWrap(True);layout.addWidget(self.audit_label)
        self.audit_table=DataTable();self.audit_table.setMinimumHeight(240);layout.addWidget(self.audit_table)
        self.audit_text=QPlainTextEdit();self.audit_text.setReadOnly(True);self.audit_text.setMaximumHeight(80);layout.addWidget(self.audit_text)
        self.export_button=QPushButton("生成 Markdown / JSON 验证报告");layout.addWidget(self.export_button)
        self.tabs.addTab(widget,"时间泄漏审计")
        self.export_button.clicked.connect(self.export_report)

    def refresh(self):
        if self.task.busy:return
        current=self.selector.currentData()
        self.records=self.context.datasets.datasets()
        self.selector.blockSignals(True);self.selector.clear()
        for record in self.records:
            self.selector.addItem(f"{record['dataset_name']} [{record['dataset_type']}]",record["id"])
        index=self.selector.findData(current)
        self.selector.setCurrentIndex(max(0,index) if self.records else -1)
        self.selector.blockSignals(False)
        if current!=self.selector.currentData():self.selection_changed()
        self.set_busy(False)

    def selection_changed(self):
        self.engine=None;self.query_result=None
        for card in self.cards:card.value_label.setText("—")
        self.summary.setText("尚未加载所选数据集的历史缓存")
        self.query_table.set_frame(pd.DataFrame());self.query_details.clear()
        self.entity_table.set_frame(pd.DataFrame());self.audit_table.set_frame(pd.DataFrame());self.audit_text.clear();self.audit_label.setText("尚未运行所选数据集审计")
        self.charts.show_history({})
        record=next((r for r in self.records if r["id"]==self.selector.currentData()),None)
        if record and record["dataset_type"]!="HISTORICAL_LABELED":
            self.status.setText("该数据集不包含真实出发延误标签，无法构建历史运行统计。")
        else:
            detected=False
            if record:
                engine=HistoricalFeatureEngine(self.context)
                detected=engine.cache_status(record["id"])=="有效历史缓存"
            self.status.setText("检测到已有历史统计缓存；点击刷新直接加载，或选择重新构建。" if detected else "请选择构建历史统计；不会自动重扫 CSV。")
        self.set_busy(False)

    def set_busy(self,busy):
        record=next((r for r in self.records if r["id"]==self.selector.currentData()),None)
        eligible=record is not None and record["dataset_type"]=="HISTORICAL_LABELED"
        self.selector.setEnabled(not busy)
        for button in [self.build_button,self.rebuild_button,self.refresh_button]:button.setEnabled(not busy and eligible)
        for button in [self.audit_button,self.query_button,self.entity_button,self.export_button,self.report_button]:button.setEnabled(not busy and self.engine is not None)
        self.cancel_button.setEnabled(busy)
        self.entity_table.setEnabled(not busy)
        if busy:
            self.started=time.perf_counter();self.timer.start()
        else:self.timer.stop()

    def set_stage(self,text):
        self.stage_text=text;self.update_status()

    def update_status(self):
        elapsed=time.perf_counter()-self.started if self.started else 0
        self.status.setText(f"{self.stage_text} · 已扫描 {self.scanned:,} 行 · 耗时 {elapsed:.1f} 秒")

    def on_progress(self,value,rows):
        self.progress.setValue(value);self.scanned=rows;self.update_status()

    def on_error(self,message):
        self.set_stage("操作失败："+message)
        QMessageBox.warning(self,"历史统计操作未完成",message)

    def build(self,force=False):
        identity=self.selector.currentData()
        if identity is None or self.task.busy:return
        engine=HistoricalFeatureEngine(self.context)
        self.progress.setValue(0);self.scanned=0
        def work(progress,cancel,stage):
            engine.build_history_cache(identity,force,progress,cancel,stage)
            return engine
        self.task.start(work,"正在准备历史缓存",self.display_engine,with_stage=True)

    def load_cache(self):
        identity=self.selector.currentData()
        if identity is None:return
        engine=HistoricalFeatureEngine(self.context)
        def work(progress,cancel):
            if not engine.load_cache(identity):raise ValueError(engine.cache_message+"；请构建或重新构建")
            return engine
        self.task.start(work,"正在检查历史缓存",self.display_engine)

    def display_engine(self,engine):
        self.engine=engine;m=engine.metadata
        values=[m["source_rows"],m.get("valid_history_rows",0),f"{m['source_min_date'] or '—'} 至 {m['source_max_date'] or '—'}",m["entity_counts"]["airline"],f"{m['entity_counts']['origin']} / {m['entity_counts']['destination']}",m["entity_counts"]["route"]]
        for card,value in zip(self.cards,values):card.value_label.setText(str(value))
        rate=m.get("delay_rows",0)/m["valid_history_rows"] if m.get("valid_history_rows") else None
        self.summary.setText(("DEMO 软件测试数据，非论文真实实验数据。 " if m["is_demo"] else "")+f"全数据集描述性延误率：{percent(rate)} · 特征 V{m['historical_feature_schema_version']} / 引擎 {m['engine_version']}\n缓存生成：{m['generated_at']} · EXPANDING · 最小历史样本：{m['min_entity_history']}")
        self.charts.show_history(m["trends"])
        self.display_audit(m["validation"])
        self.progress.setValue(100);self.set_stage("历史缓存已就绪，查询严格使用 date < D")
        self.set_busy(False)

    def run_audit(self):
        if not self.engine:return
        engine=self.engine
        self.task.start(lambda progress,cancel:engine.validation_report(progress,cancel),"正在运行泄漏审计",self.audit_ready)

    def audit_ready(self,report):
        self.progress.setValue(100)
        self.engine.metadata["validation"]=report
        self.display_audit(report)
        self.tabs.setCurrentIndex(3)
        self.set_stage("时间泄漏审计通过" if report["audit"]["passed"] else "时间泄漏审计失败")

    def display_audit(self,report):
        audit=report["audit"]
        self.audit_label.setText(f"当前自动审计{'通过' if audit['passed'] else '未通过'} · {audit['mode']} · 已检查 {audit['rows_checked']:,} 行 · 首日无历史 {report['first_date_without_history']} 行")
        self.audit_table.set_frame(pd.DataFrame([{"检查项目":item["name"],"状态":"PASS" if item["passed"] else "FAIL","说明":item["description"]} for item in audit.get("check_results",[])]))
        for i,width in enumerate([185,80,720]):self.audit_table.setColumnWidth(i,width)
        readable={"审计":audit,"冷启动数量":report["cold_start_counts"],"回退来源次数":report["fallback_source_frequency"],"无效目标日期（显式跳过）":report["invalid_target_dates_skipped"],"说明":"检查每层原始/回退来源日期 < 目标日期，输出不含后验原始字段；同日/未来标签变形测试另由自动测试覆盖。"}
        self.audit_text.setPlainText(json.dumps(readable,ensure_ascii=False,indent=2))

    def load_entities(self):
        if not self.engine or self.task.busy:return
        entity=self.entity_selector.currentData();minimum=self.minimum.value();order=self.sort_selector.currentData()
        engine=self.engine
        self.task.start(lambda progress,cancel:engine.entity_summary(entity,minimum,order).head(500),"读取实体描述性统计",self.entities_ready)

    def entities_ready(self,frame):
        self.entity_rows=frame
        display=frame.drop(columns="entity_key").rename(columns={"entity":"实体","history_count":"历史航班数","history_delay_count":"历史延误数","delay_rate":"延误率","mean_dep_delay":"平均出发延误"})
        display["延误率"]=display["延误率"].map(percent)
        display["平均出发延误"]=display["平均出发延误"].round(3)
        self.entity_table.set_frame(display)
        self.entity_table.selectionModel().currentRowChanged.connect(self.select_entity)
        if len(frame):self.entity_table.selectRow(0)

    def select_entity(self,current,previous):
        if not current.isValid() or self.task.busy:return
        row=self.entity_rows.iloc[current.row()];entity=self.entity_selector.currentData();engine=self.engine
        self.task.start(lambda progress,cancel:engine.entity_trend(entity,row.entity_key),"读取实体趋势",lambda frame:self.entity_chart.show_entity(frame,str(row.entity)))

    def query_history(self):
        if not self.engine:return
        record={"flight_date":self.query_date.date().toString("yyyy-MM-dd"),"airline":self.query_airline.text(),"origin":self.query_origin.text(),"destination":self.query_destination.text()}
        engine=self.engine
        self.task.start(lambda progress,cancel:engine.transform_single(record),"正在执行严格 as-of 查询",self.query_ready)

    def query_ready(self,result):
        self.query_result=result
        rows=[]
        for entity in ENTITIES:
            prefix=f"hist_{entity}"
            source=result.metadata[prefix+"_effective_max_source_date"]
            rows.append({"实体":ENTITY_NAMES[entity],"自身数量":result.features[prefix+"_count"],"延误率":percent(result.features[prefix+"_delay_rate"]) if pd.notna(result.features[prefix+"_delay_rate"]) else "暂无历史", "平均延误":round(result.features[prefix+"_mean_dep_delay"],3) if pd.notna(result.features[prefix+"_mean_dep_delay"]) else "暂无历史","可用":result.features[prefix+"_available"],"冷启动":result.features["cold_start_"+entity],"回退来源":result.metadata[prefix+"_fallback_source"],"最晚来源日期":source.strftime("%Y-%m-%d") if pd.notna(source) else "无历史"})
        self.query_table.set_frame(pd.DataFrame(rows))
        for i,width in enumerate([115,85,105,110,65,70,205,130]):self.query_table.setColumnWidth(i,width)
        latest=result.metadata["history_max_source_date"]
        target=result.metadata["target_date"].strftime("%Y-%m-%d")
        boundary=f"最大历史来源 {latest.strftime('%Y-%m-%d')} < 目标日期 {target}" if pd.notna(latest) else f"目标日期 {target} 之前暂无历史；NaN 不填充为 0"
        self.query_notice.setText("所有历史统计仅使用目标日期之前的航班数据（date < target date），不使用目标日期当天或未来标签。\n"+boundary+"。自身数量保持原值，展示率/均值为回退后有效值。")
        self.query_details.setPlainText(json.dumps(json_safe({"原始率与均值":{k:v for k,v in result.features.items() if k.endswith("_raw")},"历史状态":result.cold_start_info,"warnings":result.warnings}),ensure_ascii=False,indent=2))
        self.set_stage("查询完成：所有来源早于 "+result.metadata["target_date"].strftime("%Y-%m-%d"))

    def export_report(self):
        if not self.engine:return
        path,_=QFileDialog.getSaveFileName(self,"保存历史验证报告",str(self.context.root/"output/reports/stage3_history_validation.md"),"Markdown (*.md);;JSON (*.json)")
        if path:
            engine=self.engine
            self.task.start(lambda progress,cancel:export_history_report(engine,Path(path)),"正在导出历史报告",lambda result:self.set_stage("报告已保存："+result))
