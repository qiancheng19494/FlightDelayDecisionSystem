"""机场运行分析页面，第一阶段框架。"""
from app.pages.base_page import PlaceholderPage

class AirportAnalysisPage(PlaceholderPage):
    def __init__(self, context):
        super().__init__('机场运行分析', '规划：始发 / 目的机场航班量、延误率、繁忙时段与历史趋势。', context)
