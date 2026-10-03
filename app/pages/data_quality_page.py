"""数据质量报告与两种真实统计图，分析通过后台任务完成。"""
import pandas as pd
from PySide6.QtWidgets import QHBoxLayout, QGridLayout, QComboBox, QPushButton, QLabel, QProgressBar, QTabWidget, QPlainTextEdit, QFileDialog, QMessageBox
from app.pages.base_page import BasePage
from app.pages.data_manager_page import percent
from app.widgets.data_table import DataTable
from app.widgets.metric_card import MetricCard
from app.widgets.chart_widget import ChartWidget
from app.workers.task_worker import TaskController

class DataQualityPage(BasePage):
    def __init__(self, context):
        super().__init__("数据质量分析", "按数据集分析；异常保留并报告。DEMO 统计仅用于软件测试，不代表论文结果。")
        self.body.setSpacing(12)
        self.context, self.service = context, context.datasets
        self.report = None
        self.report_id = None
        self.task = TaskController(self)
        self.task.progress.connect(self.on_progress)
        self.task.status.connect(self.status_text)
        self.task.failed.connect(self.on_error)
        self.task.busy_changed.connect(self.set_busy)
        operations = QHBoxLayout()
        self.selector = QComboBox()
        self.selector.setMinimumWidth(0)
        self.selector.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.selector.setMinimumContentsLength(30)
        self.analyze_button = QPushButton("开始分析")
        self.export_button = QPushButton("导出质量报告 JSON")
        operations.addWidget(self.selector, 1)
        operations.addWidget(self.analyze_button)
        operations.addWidget(self.export_button)
        self.body.addLayout(operations)
        self.status = QLabel("请选择数据集；分析优先复用未失效的缓存。")
        self.status.setWordWrap(True)
        self.body.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.body.addWidget(self.progress)
        grid = QGridLayout()
        self.cards = []
        for i, title in enumerate(["总记录数", "字段数", "日期范围", "延误率", "取消率", "重复记录"]):
            card = MetricCard(title, "—")
            card.value_label.setObjectName("compactMetricValue")
            card.setMaximumHeight(100)
            grid.addWidget(card, i//3, i%3)
            self.cards.append(card)
        self.body.addLayout(grid)
        self.tabs = QTabWidget()
        self.fields = DataTable()
        self.charts = ChartWidget()
        self.issues = QPlainTextEdit()
        self.issues.setReadOnly(True)
        self.tabs.addTab(self.fields, "字段质量")
        self.tabs.addTab(self.charts, "标签与月度图表")
        self.tabs.addTab(self.issues, "问题摘要与数值统计")
        self.tabs.setMinimumHeight(320)
        self.body.addWidget(self.tabs)
        self.analyze_button.clicked.connect(self.start_analysis)
        self.export_button.clicked.connect(self.export_report)
        self.selector.currentIndexChanged.connect(self.selection_changed)
        self.refresh()

    def status_text(self, text):
        self.status.setText(text)

    def on_progress(self, value, rows):
        self.progress.setValue(value)
        self.status.setText(f"数据质量分析 · 已扫描 {rows:,} 行")

    def on_error(self, message):
        self.status.setText("分析失败：" + message)
        QMessageBox.warning(self, "数据质量分析未完成", message)

    def set_busy(self, busy):
        self.selector.setEnabled(not busy)
        self.analyze_button.setEnabled(not busy and self.selector.count() > 0)
        self.export_button.setEnabled(not busy and self.report is not None)

    def refresh(self):
        if self.task.busy:
            return
        old = self.selector.currentData()
        self.selector.blockSignals(True)
        self.selector.clear()
        for record in self.service.datasets():
            self.selector.addItem(f"{record['dataset_name']} [{record['status']}]", record["id"])
        index = self.selector.findData(self.service.selected_id or old)
        self.selector.setCurrentIndex(max(index, 0) if self.selector.count() else -1)
        self.selector.blockSignals(False)
        self.selection_changed()
        self.set_busy(False)

    def selection_changed(self):
        identity = self.selector.currentData()
        stale = identity is not None and self.service.source_status(self.service.repository.get(identity)) != "READY"
        if identity != self.report_id or stale:
            self.report = None
            self.fields.set_frame(pd.DataFrame())
            self.charts.show_report(None)
            self.issues.clear()
            for card in self.cards:
                card.value_label.setText("—")
            self.status.setText("请选择“开始分析”；源文件已变更时重新扫描。")
        self.set_busy(self.task.busy)

    def start_analysis(self):
        identity = self.selector.currentData()
        if identity is None or self.task.busy:
            return
        self.progress.setValue(0)
        self.task.start(lambda progress, cancel: self.service.analyze(identity, progress, cancel), "正在分析数据质量…", lambda report: self.show_report(identity, report))

    def show_report(self, identity, report):
        self.report_id, self.report = identity, report
        self.progress.setValue(100)
        self.status.setText(("DEMO / 软件测试数据，非论文实验数据 · " if report.get("is_demo") else "") + "分析完成；标签不包含已取消航班，缺失/无效取消状态的处理详见问题摘要。")
        values = [f"{report['row_count']:,}", report["column_count"], f"{report['start_date'] or '—'} 至 {report['end_date'] or '—'}", percent(report["delay_rate"]), percent(report["cancelled_rate"]), report["duplicate_rows"]]
        for card, value in zip(self.cards, values):
            card.value_label.setText(str(value))
            card.value_label.setWordWrap(True)
        fields = [{"字段名": f["field"], "标准字段": f["standard_field"], "类型（推断）": f["dtype"], "非空数量": f["non_null"], "缺失数量": f["missing"], "缺失率": percent(f["missing_rate"])} for f in report["fields"]]
        self.fields.set_frame(pd.DataFrame(fields))
        issues = [f"{name}：{count}" for name,count in report["issues"].items() if count]
        if report["duplicate_rows"]:
            issues.append(f"重复记录：{report['duplicate_rows']}")
        if report["empty_rows"]:
            issues.append(f"完全空行：{report['empty_rows']}")
        issues += report["warnings"]
        if not issues:
            issues = ["当前基础质量检查未发现严重结构问题。"]
        issues += [f"有效日期：{report['valid_dates']}；无效日期：{report['invalid_dates']}", f"可标注：{report['valid_labeled_count']}；正常：{report['normal_count']}；延误：{report['delay_count']}", f"航司：{report['airline_count']}；始发机场：{report['origin_count']}；目的机场：{report['destination_count']}", f"2400 时间值：{report['time_2400_count']}（按次日 00:00 解释，不修改原始 CSV）", "DEP_DELAY 数值统计（全部有限值，含取消行）：", str(report["delay_statistics"]), "重复定义：解码后的所有原始字段完全一致，统计第一次之后的出现次数（SHA-256）。"]
        self.issues.setPlainText("\n".join(issues))
        self.charts.show_report(report)
        self.set_busy(False)

    def export_report(self):
        if not self.report:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存数据质量报告", str(self.context.root / self.context.config.get("export_directory") / f"dataset_{self.report_id}_quality.json"), "JSON (*.json)")
        if path:
            report = self.report
            self.task.start(lambda progress, cancel: self.service.export_quality(report, path), "正在导出报告…", lambda result: self.status.setText("质量报告已保存：" + result))
