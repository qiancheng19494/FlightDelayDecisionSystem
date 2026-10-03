"""版本化特征契约；稳定 A → T → H → C 顺序，来源与特征值分离。"""
from dataclasses import dataclass, asdict
from enum import Enum
from itertools import combinations
from core.history_schema import ENTITIES
FEATURE_SCHEMA_VERSION = 1
class FeatureGroup(str, Enum):
    A="A"
    T="T"
    H="H"
    C="C"
@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    group: str
    source_fields: tuple
    description: str
    dtype: str="numeric"
    is_model_feature: bool=True
    requires_history: bool=False
    requires_schedule_context: bool=False
BASE_FEATURES_A=("airline","origin","destination","scheduled_departure","scheduled_arrival","scheduled_elapsed_time","distance")
TEMPORAL_FEATURES=("dep_hour","arr_hour","day_of_week","month","day_of_month","is_weekend","time_of_day","dep_hour_sin","dep_hour_cos","day_of_week_sin","day_of_week_cos","month_sin","month_cos","dep_time_missing","arr_time_missing","flight_date_missing")
HISTORY_FEATURES=tuple(f"hist_{e}_{m}" for e in ENTITIES for m in ("count","delay_count","delay_rate_raw","mean_dep_delay_raw","delay_rate","mean_dep_delay","available"))+tuple(f"cold_start_{e}" for e in ENTITIES)+("cold_start_any","global_history_available")
CONGESTION_KEYS={
 "origin_hour_scheduled_departures":("flight_date","origin","dep_hour"),
 "destination_hour_scheduled_arrivals":("flight_date","destination","arr_hour"),
 "airline_hour_scheduled_departures":("flight_date","airline","dep_hour"),
 "route_day_scheduled_flights":("flight_date","origin","destination"),
 "origin_day_scheduled_departures":("flight_date","origin"),
 "destination_day_scheduled_arrivals":("flight_date","destination"),
}
CONGESTION_FEATURES=tuple(CONGESTION_KEYS)+("congestion_context_available",)
CONGESTION_ALLOWED_SOURCE_FIELDS=frozenset(("flight_date",)+BASE_FEATURES_A)
GROUP_FIELDS={"A":BASE_FEATURES_A,"T":TEMPORAL_FEATURES,"H":HISTORY_FEATURES,"C":CONGESTION_FEATURES}
GROUP_DESCRIPTIONS={"A":"基础计划字段；时刻为午夜后分钟，2400=1440","T":"仅由计划日期/时间派生","H":"HistoricalFeatureEngine；过去结果 date < D","C":"目标日已知完整计划表流量代理；不是实时交通量"}
GROUP_SCHEMA={g:{"schema_version":FEATURE_SCHEMA_VERSION,"description":GROUP_DESCRIPTIONS[g],"source_type":{"A":"PLANNED","T":"PLANNED_DERIVED","H":"PAST_ONLY_DERIVED","C":"KNOWN_SCHEDULE_DERIVED"}[g],"fields":fields} for g,fields in GROUP_FIELDS.items()}
def definitions():
    result={}
    for g,fields in GROUP_FIELDS.items():
        for name in fields:
            if g=="A":src=(name,)
            elif g=="T":src=("scheduled_departure",) if name.startswith("dep_") or name=="time_of_day" else (("scheduled_arrival",) if name.startswith("arr_") else ("flight_date",))
            elif g=="H":src=("flight_date","airline","origin","destination","historical:departure_delay","historical:cancelled")
            else:src=tuple({"dep_hour":"scheduled_departure","arr_hour":"scheduled_arrival"}.get(k,k) for k in CONGESTION_KEYS.get(name,("flight_date",)))
            dtype="categorical" if name in ("airline","origin","destination","time_of_day") else "numeric"
            description=GROUP_DESCRIPTIONS[g]
            if g=="A":description={"airline":"计划航司代码，去空格并大写","origin":"计划始发机场代码","destination":"计划目的机场代码","scheduled_departure":"计划起飞时刻：午夜后分钟，2400=1440","scheduled_arrival":"计划到达时刻：午夜后分钟，2400=1440","scheduled_elapsed_time":"源计划飞行时长（分钟），不由钟面差覆盖","distance":"计划距离（沿用输入单位，BTS 为英里）"}[name]
            elif g=="C":description=("同一 "+" + ".join(CONGESTION_KEYS[name])+" 的计划表行数，含最终取消行") if name in CONGESTION_KEYS else "所提供计划上下文是否覆盖目标 flight_date；不证明数据集完整性"
            elif g=="T":description=name+"：仅计划日期/时刻；星期一=0，月份周期从 1 月=0 起算"
            elif g=="H":description=name+"：date < D 历史引擎派生；count 不因回退改变"
            result[name]=FeatureDefinition(name,g,src,description,dtype,True,g=="H",g=="C")
    return result
FEATURE_DEFINITIONS=definitions()
@dataclass(frozen=True)
class FeatureCombination:
    combination_id: str
    name: str
    groups: tuple
    feature_names: tuple
    schema_version: int=FEATURE_SCHEMA_VERSION
    description: str="通用 subset framework；论文实际组合需后续确认；不代表模型实验"
class FeatureCombinationRegistry:
    def __init__(self):
        self._items={}
        for size in range(5):
            for groups in combinations(GROUP_FIELDS,size):
                key="COMBO_"+("".join(groups) or "EMPTY")
                self._items[key]=FeatureCombination(key,"+".join(groups) or "空组合（仅框架基线）",groups,tuple(f for g in groups for f in GROUP_FIELDS[g]))
    def get_combination(self,combination="COMBO_ATHC"):
        if isinstance(combination,FeatureCombination):
            expected=self._items.get(combination.combination_id)
            if expected!=combination:raise ValueError("组合内容与注册定义不一致")
            return expected
        try:return self._items[combination]
        except (KeyError,TypeError):raise ValueError("未注册的特征组合") from None
    def list_combinations(self):return list(self._items.values())
    def validate_combination(self,combination):
        c=self.get_combination(combination)
        if len(c.feature_names)!=len(set(c.feature_names)):raise ValueError("组合包含重复字段")
        return c
