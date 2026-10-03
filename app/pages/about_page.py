"""软件说明和主要技术环境。"""
from app.pages.base_page import BasePage
from app.constants import WINDOW_TITLE

class AboutPage(BasePage):
    def __init__(self, context):
        super().__init__("软件说明 / 关于", WINDOW_TITLE)
        self.panel("软件定位", "用于航班运行 CSV 数据管理、质量分析、历史统计、特征处理和模型工程管理的本地桌面软件。")
        self.panel("当前主要功能", "CSV 后台导入、字段映射、分页浏览、BTS 兼容检查与质量报告；历史运行统计、历史特征查询与时间泄漏检查；A/T/H/C 特征处理、组合验证与样本导出；Random Forest / XGBoost 模型训练、加载、信息查看及归档。")
        self.panel("主要技术环境", "Python 3.11 / PySide6 / SQLite / JSON / logging\n数据与图表：pandas、NumPy、matplotlib；模型工程：scikit-learn、XGBoost、joblib。")
        self.panel("数据与模型使用", "历史特征仅使用目标日期之前的数据，特征检查核对字段及来源。模型管理显示资产身份、验证指标与加载状态，用户可结合数据范围查看。")
        self.panel("本地文件与依赖", "CSV、缓存、模型、报告与日志保存在本机。第三方库通过依赖环境安装，保留各自许可。")
        self.body.addStretch()
