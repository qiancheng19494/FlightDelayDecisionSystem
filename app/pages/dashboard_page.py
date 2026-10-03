"""真实空状态首页，不生成模拟业务指标。"""
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QPushButton, QLabel
from app.pages.base_page import BasePage
from app.widgets.metric_card import MetricCard

class DashboardPage(BasePage):
    def __init__(self, context, navigate):
        super().__init__("综合看板", "离线航班运行数据分析 · 数据质量、历史统计、特征与模型工程")
        self.context = context
        self.navigate = navigate
        self.content_start = self.body.count()
        self.populate()

    def populate(self):
        context, navigate = self.context, self.navigate
        summary = context.database.summary()
        current = context.datasets.current()
        report = context.datasets.cached(current) if current else None
        grid = QGridLayout()
        cards = [("已登记航班", f"{summary['flights']:,}", "已导入数据记录"), ("数据文件", summary['datasets'], "SQLite 登记数量"), ("预测记录", summary['predictions'], "已保存预测数量"), ("分类阈值", f"{context.config.get('default_threshold'):.2f}", "当前配置；尚未用于预测")]
        for i, (title, value, detail) in enumerate(cards):
            grid.addWidget(MetricCard(title, value, detail), 0, i)
        self.body.addLayout(grid)
        self.panel("当前数据集", (current["dataset_name"] + " · " + current["status"] + ("\nDEMO 为软件测试数据，不是论文真实实验数据。" if current["is_demo"] else "")) if current else "尚未导入航班数据。请在航班数据管理页导入 CSV。")
        row = QHBoxLayout()
        for title, index in [("查看数据管理", 1), ("查看模型状态", 6), ("软件说明", 14)]:
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, target=index: navigate(target))
            row.addWidget(button)
        row.addStretch()
        self.body.addLayout(row)
        from app.pages.data_manager_page import percent
        if report:
            self.panel("当前数据集统计", f"记录：{report['row_count']:,} · 日期：{report['start_date']} 至 {report['end_date']}\n航司：{report['airline_count']} · 始发机场：{report['origin_count']} · 目的机场：{report['destination_count']}\n有效标签延误率：{percent(report['delay_rate'])} · 取消：{report['cancelled_count']}；统计直接读取缓存。")
        else:
            self.panel("数据概况", "暂无有效缓存统计。源文件移动或修改后请到数据质量页重新分析。")
        datasets = context.database.recent("dataset_records")
        predictions = context.database.recent("prediction_records")
        self.panel("最近活动", "最近导入：" + ("、".join(row['file_name'] for row in datasets) or "暂无导入记录") + "\n最近预测：" + ("、".join(str(row['id']) for row in predictions) or "暂无预测记录"))
        self.panel("系统状态", f"数据库：已连接  ·  配置：已加载  ·  日志：已启用\n默认模型偏好：{context.config.get('default_model')}  ·  模型状态：请查看模型管理\n模型加载状态以模型管理页为准")
        if context.config.warnings:
            self.panel("配置提示", "\n".join(context.config.warnings))
        self.body.addStretch()

    def refresh(self):
        """只查询元信息和缓存，不重读航班 CSV。"""
        def clear(layout, start=0):
            while layout.count() > start:
                item = layout.takeAt(start)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    clear(item.layout())
        clear(self.body, self.content_start)
        self.populate()
