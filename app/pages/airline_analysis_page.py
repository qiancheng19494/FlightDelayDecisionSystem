"""航司运行分析页面，第一阶段框架。"""
from app.pages.base_page import PlaceholderPage

class AirlineAnalysisPage(PlaceholderPage):
    def __init__(self, context):
        super().__init__('航司运行分析', '规划：航司航班量、延误率、平均延误及月度和小时趋势。', context)
