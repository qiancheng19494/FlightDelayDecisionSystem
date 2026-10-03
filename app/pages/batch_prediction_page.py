"""批量延误预测页面，第一阶段框架。"""
from app.pages.base_page import PlaceholderPage

class BatchPredictionPage(PlaceholderPage):
    def __init__(self, context):
        super().__init__('批量延误预测', '当前版本已具备模型工程能力，本页面尚未接入批量预测工作流。', context)
