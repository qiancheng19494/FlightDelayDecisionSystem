"""后验字段可用于历史分析和标签，但禁止直接作为目标航班预测输入。"""
from core.field_mapper import FieldMapper
PRE_FLIGHT_ALLOWED_FIELDS = frozenset({"flight_date", "airline", "origin", "destination", "scheduled_departure", "scheduled_arrival", "scheduled_elapsed_time", "distance"})
POST_EVENT_FORBIDDEN_FIELDS = frozenset({"departure_delay", "arrival_delay", "actual_departure_time", "actual_arrival_time", "wheels_off", "wheels_on", "air_time", "taxi_out", "taxi_in", "actual_elapsed_time", "cancelled", "diverted", "delay_label"})

def validate_prediction_fields(fields: list[str]) -> list[str]:
    """采用白名单；标准名、BTS 别名和未知字段均不能绕过策略。"""
    mapper = FieldMapper()
    canonical = [mapper.canonical(name) or name for name in fields]
    rejected = [name for name in canonical if name not in PRE_FLIGHT_ALLOWED_FIELDS]
    if rejected:
        raise ValueError("禁止或未经批准的预测字段：" + ", ".join(rejected))
    return canonical

# 字段名批准不等同于来源验证；历史字段必须由 HistoricalFeatureEngine 产生。
from core.history_schema import HISTORY_FIELDS
HISTORICAL_DERIVED_FIELDS = HISTORY_FIELDS
HISTORICAL_FIELD_PROVENANCE = "PAST_ONLY_DERIVED"
# 第四阶段注册字段；来源契约另由 FeatureValidator 核验。
from core.feature_groups import TEMPORAL_FEATURES, CONGESTION_FEATURES
TIME_DERIVED_FIELDS = frozenset(TEMPORAL_FEATURES)
PLANNED_CONGESTION_FIELDS = frozenset(CONGESTION_FEATURES)

def validate_model_feature_list(fields: list[str]) -> list[str]:
    """统一入口允许已注册的计划/历史/时间/拥堵字段，拒绝后验与来源字符串。"""
    mapper=FieldMapper()
    names=[mapper.canonical(name) or name for name in fields]
    allowed=PRE_FLIGHT_ALLOWED_FIELDS | HISTORICAL_DERIVED_FIELDS | TIME_DERIVED_FIELDS | PLANNED_CONGESTION_FIELDS
    rejected=[name for name in names if name not in allowed or name in POST_EVENT_FORBIDDEN_FIELDS]
    if rejected:
        raise ValueError("禁止或未经批准的模型字段："+", ".join(rejected))
    return names
