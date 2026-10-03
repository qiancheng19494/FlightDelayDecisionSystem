"""单航班延误预测页面，第一阶段框架。"""
from app.pages.base_page import PlaceholderPage

class SinglePredictionPage(PlaceholderPage):
    def __init__(self, context):
        super().__init__('单航班延误预测', '模型可在模型性能分析的模型管理页查看和加载。本页面尚未接入单航班预测工作流。', context)
