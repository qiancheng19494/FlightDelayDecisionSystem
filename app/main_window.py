"""桌面主窗口，仅负责布局与导航。"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QListWidget, QStackedWidget, QScrollArea, QMessageBox
from app.constants import WINDOW_TITLE
from app.page_registry import PAGE_FACTORIES
from app.pages.dashboard_page import DashboardPage
from app.pages.about_page import AboutPage

class MainWindow(QMainWindow):
    def __init__(self, context):
        super().__init__()
        self.context = context
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(context.config.get("window_width"), context.config.get("window_height"))
        self.setMinimumSize(1000, 640)
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        sidebar = QWidget(objectName="sidebar")
        sidebar.setFixedWidth(220)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 24, 14, 20)
        side.addWidget(QLabel("FLIGHT DELAY", objectName="brand"))
        side.addWidget(QLabel("数据分析 · 延误预测", objectName="sideSubtitle"))
        self.navigation = QListWidget(objectName="navigation")
        self.navigation.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.navigation.addItems(["综合看板"] + [title for title, _ in PAGE_FACTORIES] + ["软件说明 / 关于"])
        side.addWidget(self.navigation)
        side.addWidget(QLabel("V1.0  /  离线桌面系统", objectName="sideSubtitle"))
        outer.addWidget(sidebar)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        header = QLabel(WINDOW_TITLE, objectName="header")
        header.setWordWrap(True)
        right.addWidget(header)
        self.stack = QStackedWidget()
        self.pages = [DashboardPage(context, self.navigation.setCurrentRow)] + [factory(context) for _, factory in PAGE_FACTORIES] + [AboutPage(context)]
        for page in self.pages:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
        right.addWidget(self.stack)
        outer.addLayout(right, 1)
        self.pages[1].datasets_changed.connect(self.pages[0].refresh)
        self.pages[1].datasets_changed.connect(self.pages[2].refresh)
        self.pages[1].datasets_changed.connect(self.pages[3].refresh)
        self.navigation.currentRowChanged.connect(self.change_page)
        self.navigation.setCurrentRow(0)
        self.statusBar().showMessage("就绪  |  数据库已初始化  |  模型状态请查看模型管理")

    def change_page(self, index):
        """切换页面并记录用户导航事件。"""
        if 0 <= index < self.stack.count():
            self.stack.setCurrentIndex(index)
            if index in (0, 2, 3, 6):
                self.pages[index].refresh()
            self.context.logger.debug("切换页面：%s", self.navigation.item(index).text())

    def closeEvent(self, event):
        from app.workers.task_worker import TaskController
        if any(task.busy for task in self.findChildren(TaskController)):
            QMessageBox.information(self, "任务正在运行", "请等待导入、预览或分析完成后再关闭窗口。")
            event.ignore()
            return
        self.context.logger.info("主窗口关闭")
        super().closeEvent(event)
