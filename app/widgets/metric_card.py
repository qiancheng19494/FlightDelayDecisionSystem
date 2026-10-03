"""首页指标卡片。"""
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

class MetricCard(QFrame):
    def __init__(self, title, value, detail="", parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(title, objectName="muted"))
        self.value_label = QLabel(str(value), objectName="metricValue")
        layout.addWidget(self.value_label)
        if detail:
            label = QLabel(detail, objectName="muted")
            label.setWordWrap(True)
            layout.addWidget(label)
