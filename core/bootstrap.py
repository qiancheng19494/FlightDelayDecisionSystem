"""应用基础设施装配，与 GUI 页面分离。"""
from dataclasses import dataclass
from pathlib import Path
from core.config_manager import ConfigManager
from core.logging_service import setup_logging
from database.db_manager import DatabaseManager

@dataclass
class AppContext:
    root: Path
    config: ConfigManager
    database: DatabaseManager
    logger: object
    datasets: object = None

def initialize(root):
    """首次启动创建目录、加载配置并初始化数据库。"""
    root = Path(root).resolve()
    config = ConfigManager(root)
    directories = ["data/raw", "data/processed", "data/cache", "data/demo", "logs", "output/predictions", "output/reports", "output/figures", "models", "database", config.get("data_directory"), config.get("export_directory")]
    for directory in directories:
        (root / directory).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(root / "logs", config.get("log_level"))
    for warning in config.warnings:
        logger.warning(warning)
    database = DatabaseManager(root / "database/flight_delay.sqlite3")
    database.initialize()
    database.log_event("INFO", "application_initialized", "V1.0；未加载预测模型")
    logger.info("基础设施初始化成功；数据库已就绪；预测模型未加载")
    context = AppContext(root, config, database, logger)
    from core.dataset_service import DatasetService
    context.datasets = DatasetService(context)
    return context
