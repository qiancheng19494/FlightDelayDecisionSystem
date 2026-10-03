"""集中注册导航，业务页面各自独立。"""
from app.pages.data_manager_page import DataManagerPage
from app.pages.data_quality_page import DataQualityPage
from app.pages.history_analysis_page import HistoryAnalysisPage
from app.pages.single_prediction_page import SinglePredictionPage
from app.pages.batch_prediction_page import BatchPredictionPage
from app.pages.model_evaluation_page import ModelEvaluationPage
from app.pages.threshold_analysis_page import ThresholdAnalysisPage
from app.pages.shap_analysis_page import ShapAnalysisPage
from app.pages.airline_analysis_page import AirlineAnalysisPage
from app.pages.airport_analysis_page import AirportAnalysisPage
from app.pages.prediction_records_page import PredictionRecordsPage
from app.pages.export_page import ExportPage
from app.pages.settings_page import SettingsPage
PAGE_FACTORIES = [
    ('航班数据管理', DataManagerPage),
    ('数据质量分析', DataQualityPage),
    ('历史运行统计', HistoryAnalysisPage),
    ('单航班延误预测', SinglePredictionPage),
    ('批量延误预测', BatchPredictionPage),
    ('模型性能分析', ModelEvaluationPage),
    ('阈值决策分析', ThresholdAnalysisPage),
    ('特征解释分析', ShapAnalysisPage),
    ('航司运行分析', AirlineAnalysisPage),
    ('机场运行分析', AirportAnalysisPage),
    ('预测记录管理', PredictionRecordsPage),
    ('报告与结果导出', ExportPage),
    ('系统设置', SettingsPage)
]
