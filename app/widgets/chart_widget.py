"""嵌入 Qt 的真实标签与月度统计图。"""
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtWidgets import QWidget, QVBoxLayout
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

class ChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(9, 3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setMinimumHeight(280)
        self.show_report(None)

    def show_report(self, report):
        self.figure.clear()
        left, right = self.figure.subplots(1, 2)
        if report and report["valid_labeled_count"]:
            values = [report["normal_count"], report["delay_count"]]
            left.pie(values, labels=["正常", "延误"], autopct="%1.1f%%", colors=["#3a8eae", "#e6a34b"])
            left.set_title("有效标签占比（排除已知取消）")
        else:
            left.text(.5, .5, "当前数据集不包含真实延误标签。", ha="center", va="center", wrap=True, transform=left.transAxes)
            left.set_axis_off()
        monthly = report.get("monthly_summary", {}) if report else {}
        if monthly:
            # 多年月度数据按视图抽稀刻度，数值全部绘制。
            keys = list(monthly)
            right.bar(range(len(keys)), [monthly[k]["rows"] for k in keys], color="#286bad")
            stride = max(1, len(keys)//6)
            right.set_xticks(range(0, len(keys), stride), keys[::stride], rotation=25)
            right.set_title("月度航班数量（有效日期）")
            right.set_ylabel("记录数")
        else:
            right.text(.5, .5, "暂无有效日期，无法绘制月度图。", ha="center", va="center", transform=right.transAxes)
            right.set_axis_off()
        self.canvas.draw_idle()
