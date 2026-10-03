"""历史特征契约：数值特征与来源/回退元信息分开，日级严格 date < D。"""
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
ENGINE_VERSION = "1.0.1"
SCHEMA_VERSION = 1
ENTITIES = {"global": (), "airline": ("airline",), "origin": ("origin",), "destination": ("destination",), "route": ("origin", "destination"), "airline_origin": ("airline", "origin")}
DEFAULT_HISTORY_CONFIG = {"min_entity_history": 1, "window_mode": "EXPANDING", "drop_exact_duplicates": False, "enable_leakage_audit": True}

class HistoryWindowMode(str, Enum):
    EXPANDING = "EXPANDING"
    ROLLING = "ROLLING"

class HistoryStatus(str, Enum):
    NO_HISTORY = "NO_HISTORY"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    AVAILABLE = "AVAILABLE"

HISTORY_FIELDS = frozenset({f"hist_{entity}_{metric}" for entity in ENTITIES for metric in ("count", "delay_count", "delay_rate_raw", "mean_dep_delay_raw", "delay_rate", "mean_dep_delay", "available")} | {f"cold_start_{entity}" for entity in ENTITIES} | {"cold_start_any", "global_history_available"})

@dataclass
class HistoricalFeatureResult:
    features: dict
    metadata: dict
    warnings: list[str] = field(default_factory=list)
    cold_start_info: dict = field(default_factory=dict)

@dataclass
class HistoricalBatchResult:
    features: pd.DataFrame
    metadata: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    cold_start_info: dict = field(default_factory=dict)

def history_config(values: dict | None = None) -> dict:
    """未实现的窗口/去重选项显式拒绝，不静默忽略配置。"""
    config = {**DEFAULT_HISTORY_CONFIG, **(values or {})}
    if type(config["min_entity_history"]) is not int or config["min_entity_history"] < 1:
        raise ValueError("最小历史样本量必须为正整数")
    if config["window_mode"] != "EXPANDING":
        raise ValueError("第三阶段仅支持 EXPANDING；ROLLING 尚未实现")
    if config["drop_exact_duplicates"] is not False:
        raise ValueError("第三阶段保留重复记录，尚不支持自动去重")
    if type(config["enable_leakage_audit"]) is not bool:
        raise ValueError("enable_leakage_audit 必须是布尔值")
    return config
