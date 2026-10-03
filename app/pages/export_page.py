"""报告与结果导出页面，第一阶段框架。"""
from app.pages.base_page import PlaceholderPage

class ExportPage(PlaceholderPage):
    def __init__(self, context):
        super().__init__('报告与结果导出', '质量报告、历史验证报告与特征样本可在对应业务页面导出。本页面尚未接入统一报告与预测结果导出工作流。', context)
