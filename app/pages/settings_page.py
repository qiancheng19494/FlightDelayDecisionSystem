"""系统设置页面，第一阶段框架。"""
from app.pages.base_page import PlaceholderPage

class SettingsPage(PlaceholderPage):
    def __init__(self, context):
        super().__init__('系统设置', '配置基础设施已就绪，当前页面尚未提供编辑控件。可在 config/user_config.json 配置默认模型、阈值、相对目录、分页数量、日志级别和窗口尺寸；下次启动读取。模型名称仅为偏好，不表示已加载。', context)
