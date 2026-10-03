"""保留 Stage 4 公共控件/API，在同一导航页增加二级训练及管理。"""
from PySide6.QtWidgets import QWidget,QVBoxLayout,QTabWidget
from app.pages.feature_engineering_page import FeatureEngineeringPage
from app.pages.model_training_page import ModelTrainingPage,ModelManagementPage
class ModelEvaluationPage(FeatureEngineeringPage):
    def __init__(self,context):
        super().__init__(context)
        feature=QWidget();layout=QVBoxLayout(feature);layout.setContentsMargins(0,0,0,0)
        while self.body.count():
            item=self.body.takeAt(0)
            if item.widget():layout.addWidget(item.widget())
            elif item.layout():layout.addLayout(item.layout())
            else:layout.addItem(item)
        self.mode_tabs=QTabWidget();self.body.addWidget(self.mode_tabs);self.mode_tabs.addTab(feature,"特征工程")
        self.management=ModelManagementPage(context);self.training=ModelTrainingPage(context,self.management.refresh)
        self.mode_tabs.addTab(self.training,"模型训练");self.mode_tabs.addTab(self.management,"模型管理")
    def refresh(self):
        super().refresh()
        if hasattr(self,"training"):self.training.refresh();self.management.refresh()
