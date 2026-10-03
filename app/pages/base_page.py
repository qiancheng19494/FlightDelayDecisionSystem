"""统一页面结构与明确的未实现状态。"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame

class BasePage(QWidget):
    def __init__(self, title, subtitle, parent=None):
        super().__init__(parent)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(30, 24, 30, 24)
        self.body.setSpacing(18)
        self.body.addWidget(QLabel(title, objectName="pageTitle"))
        label = QLabel(subtitle, objectName="muted")
        label.setWordWrap(True)
        self.body.addWidget(label)

    def panel(self, title, text):
        frame = QFrame(objectName="panel")
        layout = QVBoxLayout(frame)
        layout.addWidget(QLabel(title, objectName="sectionTitle"))
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(label)
        self.body.addWidget(frame)
        return frame

class PlaceholderPage(BasePage):
    def __init__(self, title, description, context):
        super().__init__(title, "预留功能入口")
        self.panel("功能尚未接入", description)
        self.panel("当前状态", "此页面仅提供功能说明，当前版本不执行对应业务操作。")
        self.body.addStretch()
