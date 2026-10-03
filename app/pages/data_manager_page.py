"""数据管理页面，仅协调用户交互，扫描和预览由后台执行。"""
import json
from pathlib import Path
import pandas as pd
from PySide6.QtCore import Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QLabel, QProgressBar, QFileDialog, QMessageBox, QPlainTextEdit
from app.pages.base_page import BasePage
from app.widgets.data_table import DataTable
from app.workers.task_worker import TaskController

def percent(value):
    return "—" if value is None else f"{value:.2%}"

class DataManagerPage(BasePage):
    datasets_changed = Signal()
    def __init__(self, context):
        super().__init__("航班数据管理", "导入并管理统计分析和待预测航班 CSV；DEMO 为软件测试数据，不是论文真实实验数据。")
        self.context, self.service = context, context.datasets
        self.records, self.selected, self.page = [], None, 1
        self.task = TaskController(self)
        self.task.progress.connect(self.on_progress)
        self.task.status.connect(self.show_status)
        self.task.failed.connect(self.on_error)
        self.task.busy_changed.connect(self.set_busy)
        self.setAcceptDrops(True)
        row = QHBoxLayout()
        self.import_button = QPushButton("导入 CSV")
        self.refresh_button = QPushButton("刷新")
        self.directory_button = QPushButton("打开数据目录")
        self.compatibility_button = QPushButton("验证 BTS 兼容性")
        for button in [self.import_button, self.refresh_button, self.directory_button, self.compatibility_button]:
            row.addWidget(button)
        row.addStretch()
        self.body.addLayout(row)
        self.import_button.clicked.connect(self.choose_file)
        self.refresh_button.clicked.connect(self.refresh)
        self.directory_button.clicked.connect(self.open_directory)
        self.compatibility_button.clicked.connect(self.validate_compatibility)
        self.status = QLabel("请选择或拖入 CSV 文件；" + ("将复制源文件到数据目录。" if context.config.get("copy_imported_files") else "只登记原始路径，不复制源文件。"))
        self.status.setWordWrap(True)
        self.body.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.body.addWidget(self.progress)
        self.dataset_table = DataTable()
        self.dataset_table.setFixedHeight(130)
        self.body.addWidget(self.dataset_table)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setFixedHeight(110)
        self.details.setPlaceholderText("选择数据集查看路径、字段映射与统计详情。")
        self.body.addWidget(self.details)
        self.preview_table = DataTable()
        self.preview_table.setMinimumHeight(200)
        self.preview_table.setMaximumHeight(260)
        self.body.addWidget(self.preview_table)
        pager = QHBoxLayout()
        self.previous_button, self.next_button = QPushButton("上一页"), QPushButton("下一页")
        self.page_label = QLabel("暂无预览")
        pager.addWidget(self.previous_button)
        pager.addWidget(self.page_label)
        pager.addWidget(self.next_button)
        pager.addStretch()
        self.body.addLayout(pager)
        self.previous_button.clicked.connect(lambda: self.load_page(self.page-1))
        self.next_button.clicked.connect(lambda: self.load_page(self.page+1))
        self.compatibility_details=QPlainTextEdit();self.compatibility_details.setReadOnly(True);self.compatibility_details.setMaximumHeight(180)
        self.compatibility_details.setPlaceholderText("选择数据集后验证 BTS 字段、时间、标签及 H/FeaturePipeline 前置条件；仅抽查前 1000 行。")
        self.body.addWidget(self.compatibility_details)
        self.refresh()

    def validate_compatibility(self):
        if self.task.busy or not self.selected:return
        if self.selected["status"]!="READY":self.status.setText("源文件失效，无法验证");return
        from core.bts_compatibility import verify
        path=self.selected["file_path"];is_demo=self.selected["is_demo"]
        def ready(result):
            self.compatibility_result=result
            summary=("DEMO / 非真实 BTS 验证" if is_demo else "所选本地数据 · 有界兼容性检查")+"\n"
            summary+=f"DatasetType：{result['mapping']['dataset_type']}；映射：{result['compatible']}；样本：{result.get('sampled_rows',0)} 行\n"
            summary+=f"时间格式：{result.get('scheduled_times')}\n标签：{result.get('labels')}\nFeaturePipeline：{result.get('feature_pipeline_compatibility')}；History：{result.get('history_compatibility')}\n"
            summary+="字段兼容不等于已通过完整 Pipeline；完整 smoke 证据见 Stage 5.2 报告。\n"
            self.compatibility_details.setPlainText(summary+json.dumps(result,ensure_ascii=False,indent=2));self.status.setText("BTS 兼容性抽样检查完成，源文件未修改。")
        self.task.start(lambda progress,cancel:verify(path,1000),"正在验证 BTS 兼容性",ready)

    def show_status(self, text):
        self.status.setText(text)

    def on_progress(self, value, rows):
        self.progress.setValue(value)
        self.status.setText(f"{getattr(self, 'active_file', '')} · 扫描 / 统计中 · 已扫描 {rows:,} 行")

    def set_busy(self, busy):
        self.import_button.setEnabled(not busy)
        self.refresh_button.setEnabled(not busy)
        self.compatibility_button.setEnabled(not busy)
        self.dataset_table.setEnabled(not busy)
        self.update_pager()

    def on_error(self, message):
        self.status.setText("操作失败：" + message)
        self.progress.setValue(0)
        QMessageBox.warning(self, "数据操作未完成", message)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择航班 CSV", str(self.context.root / self.context.config.get("data_directory")), "CSV (*.csv)")
        if path:
            self.begin_import(path)

    def begin_import(self, path):
        if self.task.busy:
            return
        self.active_file = Path(path).name
        self.progress.setValue(0)
        self.task.start(lambda progress, cancel: self.service.inspect(path), "正在识别文件和字段…", lambda info: self.confirm_import(path, info))

    def confirm_import(self, path, info):
        mapping = info["mapping"]
        description = "\n".join(f"{source} → {target}" for source, target in mapping["mapped_fields"].items())
        answer = QMessageBox.question(self, "字段识别完成", f"文件：{info['file_name']}\n类型：{mapping['dataset_type']}\n编码：{info['encoding']}\n{description}\n未知字段：{', '.join(mapping['unknown_fields']) or '无'}\n是否开始导入？")
        if answer != QMessageBox.Yes:
            self.status.setText("已取消导入，未登记数据集。")
            return
        self.task.start(lambda progress, cancel: self.service.import_dataset(path, progress, cancel), f"开始导入 {Path(path).name}", self.imported)

    def imported(self, record):
        self.progress.setValue(100)
        self.status.setText(f"导入成功：{record['dataset_name']} · {record['row_count']:,} 行 · {record['dataset_type']}")
        self.service.selected_id = record["id"]
        self.refresh(record["id"])
        self.datasets_changed.emit()

    def refresh(self, selected_id=None):
        if self.task.busy:
            return
        selected_id = selected_id if type(selected_id) is int else (self.selected["id"] if self.selected else self.service.selected_id)
        self.records = self.service.datasets()
        data = [{"数据集": r["dataset_name"], "文件名": r["file_name"], "类型": r["dataset_type"], "记录数": r["row_count"], "日期范围": f"{r['start_date'] or '—'} ~ {r['end_date'] or '—'}", "航司数": r["airline_count"], "始发/目的机场": f"{r['origin_count']}/{r['destination_count']}", "延误率": percent(r["delay_rate"]), "导入时间 UTC": r["imported_at"], "状态": r["status"]} for r in self.records]
        self.dataset_table.set_frame(pd.DataFrame(data))
        self.dataset_table.selectionModel().currentRowChanged.connect(self.selection_changed)
        if self.records:
            index = next((i for i,r in enumerate(self.records) if r["id"] == selected_id), 0)
            self.dataset_table.selectRow(index)
        else:
            self.selected = None
            self.update_pager()

    def selection_changed(self, current, previous):
        if not current.isValid():
            return
        self.selected = r = self.records[current.row()]
        self.service.selected_id = r["id"]
        mapping = json.loads(r["mapping_json"] or "{}")
        self.details.setPlainText(f"{r['dataset_name']}  |  {r['status']}\n路径：{r['file_path']}\n大小：{r['file_size']} 字节 · 编码：{r['encoding']} · 字段：{r['column_count']} · 日期：{r['start_date']} 至 {r['end_date']}\n取消：{r['cancelled_count']} · 有效标签：{r['valid_labeled_count']} · 延误：{r['delay_count']} · 延误率：{percent(r['delay_rate'])}\n字段映射：{json.dumps(mapping.get('mapped_fields', {}), ensure_ascii=False)}")
        self.page = 1
        self.preview_table.set_frame(pd.DataFrame())
        self.load_page(1)

    def load_page(self, page):
        if self.task.busy or not self.selected:
            return
        if self.selected["status"] != "READY":
            self.page_label.setText("源文件不可用或已变化；请恢复文件或到质量页重新分析。")
            self.update_pager()
            return
        self.page = page
        identity = self.selected["id"]
        self.task.start(lambda progress, cancel: self.service.preview(identity, page), "正在读取分页预览…", self.preview_ready)

    def preview_ready(self, frame):
        self.preview_table.set_frame(frame)
        self.update_pager()
        if self.selected:
            self.status.setText(f"数据集已就绪 / 导入成功：{self.selected['dataset_name']} · 共 {self.selected['row_count']:,} 行；显示原始值，异常未删除。")

    def update_pager(self):
        total = self.selected["row_count"] if self.selected else 0
        size = self.context.config.get("page_size")
        ready = self.selected and self.selected["status"] == "READY" and not self.task.busy
        self.previous_button.setEnabled(bool(ready and self.page > 1))
        self.next_button.setEnabled(bool(ready and self.page * size < total))
        if self.selected and self.selected["status"] == "READY":
            self.page_label.setText(f"第 {self.page} 页 · {(self.page-1)*size+1 if total else 0}–{min(self.page*size,total)} / {total:,}")

    def open_directory(self):
        directory = self.context.root / self.context.config.get("data_directory")
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory))):
            self.on_error("无法打开数据目录，请检查目录是否存在。")

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if not self.task.busy and len(urls) == 1 and urls[0].toLocalFile().lower().endswith(".csv"):
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.begin_import(event.mimeData().urls()[0].toLocalFile())
        event.acceptProposedAction()
