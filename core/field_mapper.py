"""确定性字段别名映射，不进行模糊匹配。"""
from enum import Enum
from dataclasses import dataclass, asdict

class DatasetType(str, Enum):
    HISTORICAL_LABELED = "HISTORICAL_LABELED"
    PREDICTION_INPUT = "PREDICTION_INPUT"
    UNKNOWN = "UNKNOWN"

ALIASES = {
    "flight_date": ["FL_DATE", "FlightDate", "FLIGHT_DATE"],
    "airline": ["OP_UNIQUE_CARRIER", "OP_CARRIER", "UNIQUE_CARRIER", "Reporting_Airline", "AIRLINE"],
    "origin": ["ORIGIN"], "destination": ["DEST", "Destination", "DESTINATION"],
    "scheduled_departure": ["CRS_DEP_TIME", "CRSDepTime"],
    "scheduled_arrival": ["CRS_ARR_TIME", "CRSArrTime"],
    "scheduled_elapsed_time": ["CRS_ELAPSED_TIME", "CRSElapsedTime"],
    "distance": ["DISTANCE"], "departure_delay": ["DEP_DELAY", "DepDelay"],
    "cancelled": ["CANCELLED"], "arrival_delay": ["ARR_DELAY", "ArrDelay"],
    "flight_number": ["OP_CARRIER_FL_NUM", "FL_NUM", "Flight_Number_Reporting_Airline"],
    "tail_number": ["TAIL_NUM", "Tail_Number"],
    "origin_city": ["ORIGIN_CITY_NAME", "OriginCityName"],
    "destination_city": ["DEST_CITY_NAME", "DestCityName"],
    "taxi_out": ["TAXI_OUT", "TaxiOut"], "taxi_in": ["TAXI_IN", "TaxiIn"],
    "wheels_off": ["WHEELS_OFF", "WheelsOff"], "wheels_on": ["WHEELS_ON", "WheelsOn"],
    "actual_elapsed_time": ["ACTUAL_ELAPSED_TIME", "ActualElapsedTime"],
    "air_time": ["AIR_TIME", "AirTime"], "diverted": ["DIVERTED"],
    "actual_departure_time": ["DEP_TIME", "DepTime"],
    "actual_arrival_time": ["ARR_TIME", "ArrTime"],
}
REQUIRED = ("flight_date", "airline", "origin", "destination")

@dataclass
class MappingResult:
    mapped_fields: dict[str, str]
    missing_required_fields: list[str]
    optional_fields: list[str]
    unknown_fields: list[str]
    conflicts: list[str]
    is_valid: bool
    dataset_type: DatasetType

    def to_dict(self) -> dict:
        return asdict(self)

class FieldMapper:
    """同一标准字段出现多个源列时拒绝导入，避免静默选错列。"""
    def canonical(self, name: str) -> str | None:
        name = str(name).strip().lstrip("\ufeff")
        for target, aliases in ALIASES.items():
            if name in [target, *aliases]:
                return target
        for target, aliases in ALIASES.items():
            if name.casefold() in {s.casefold() for s in [target, *aliases]}:
                return target
        return None

    def map_fields(self, columns: list[str]) -> MappingResult:
        mapped, unknown, conflicts = {}, [], []
        for column in columns:
            target = self.canonical(column)
            if target is None:
                unknown.append(column)
            elif target in mapped.values():
                conflicts.append(target)
            else:
                mapped[column] = target
        missing = [name for name in REQUIRED if name not in mapped.values()]
        valid = not missing and not conflicts and len(set(columns)) == len(columns)
        kind = DatasetType.UNKNOWN if not valid else (DatasetType.HISTORICAL_LABELED if "departure_delay" in mapped.values() else DatasetType.PREDICTION_INPUT)
        return MappingResult(mapped, missing, [c for c in mapped.values() if c not in REQUIRED], unknown, conflicts, valid, kind)
