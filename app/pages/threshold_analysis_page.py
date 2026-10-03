"""阈值决策分析页面，第一阶段框架。"""
from app.pages.base_page import PlaceholderPage

class ThresholdAnalysisPage(PlaceholderPage):
    def __init__(self, context):
        super().__init__('阈值决策分析', '规划：阈值调整、验证集 F1 阈值及 Precision / Recall / F1 曲线。阈值需根据当前数据分布和实际使用目标进行选择；调整阈值不重新训练模型。', context)
