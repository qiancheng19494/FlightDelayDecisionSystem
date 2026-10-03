"""特征解释分析页面，第一阶段框架。"""
from app.pages.base_page import PlaceholderPage

class ShapAnalysisPage(PlaceholderPage):
    def __init__(self, context):
        super().__init__('特征解释分析', '规划：全局重要性、SHAP summary 与单条记录贡献。SHAP 表示模型输出的解释，不代表因果关系。', context)
