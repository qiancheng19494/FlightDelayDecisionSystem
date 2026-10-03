"""仅使用本次严格 as-of 查询的过去统计回退，实体数量绝不替换。"""
import numpy as np
import pandas as pd
from core.history_schema import ENTITIES

class HistoricalFallbackPolicy:
    def apply(self, features: pd.DataFrame, metadata: pd.DataFrame) -> None:
        """原始率/均值保留；有效值只从 available 的历史层级获得。"""
        for entity in ENTITIES:
            prefix = "hist_" + entity
            available = features[prefix + "_available"].eq(1)
            for metric in ["delay_rate", "mean_dep_delay"]:
                features[prefix+"_"+metric] = features[prefix+"_"+metric+"_raw"].where(available)
            metadata[prefix+"_fallback_source"] = np.where(available, entity.upper(), "NO_HISTORY")
            metadata[prefix+"_effective_max_source_date"] = metadata[prefix+"_max_source_date"].where(available)
            if entity == "global":
                continue
            parents = ("origin", "destination") if entity == "route" else (("airline", "origin") if entity == "airline_origin" else ())
            missing = ~available
            if parents:
                first, second = parents
                a = features[f"hist_{first}_available"].eq(1)
                b = features[f"hist_{second}_available"].eq(1)
                both = missing & a & b
                for metric in ["delay_rate", "mean_dep_delay"]:
                    features.loc[both,prefix+"_"+metric] = (features.loc[both,f"hist_{first}_{metric}_raw"] + features.loc[both,f"hist_{second}_{metric}_raw"])/2
                metadata.loc[both,prefix+"_fallback_source"] = ("AIRLINE_ORIGIN_MEAN" if entity == "airline_origin" else "ORIGIN_DESTINATION")
                metadata.loc[both,prefix+"_effective_max_source_date"] = metadata.loc[both,[f"hist_{first}_max_source_date",f"hist_{second}_max_source_date"]].max(axis=1)
                for parent, mask in [(first,missing & a & ~b),(second,missing & b & ~a)]:
                    self._copy(parent, prefix, mask, features, metadata)
                missing &= ~a & ~b
            global_available = features.hist_global_available.eq(1)
            self._copy("global", prefix, missing & global_available, features, metadata)
        features["cold_start_any"] = features[[f"cold_start_{e}" for e in ENTITIES]].max(axis=1).astype(int)
        features["global_history_available"] = features.hist_global_available
        metadata["history_max_source_date"] = metadata[[f"hist_{e}_max_source_date" for e in ENTITIES]+[f"hist_{e}_effective_max_source_date" for e in ENTITIES]].max(axis=1)

    @staticmethod
    def _copy(parent, prefix, mask, features, metadata):
        for metric in ["delay_rate", "mean_dep_delay"]:
            features.loc[mask,prefix+"_"+metric] = features.loc[mask,f"hist_{parent}_{metric}_raw"]
        metadata.loc[mask,prefix+"_fallback_source"] = parent.upper()
        metadata.loc[mask,prefix+"_effective_max_source_date"] = metadata.loc[mask,f"hist_{parent}_max_source_date"]
