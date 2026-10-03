"""审计特征来源日期与字段边界；结果不替代变形测试或任意外部缓存真实性证明。"""
import pandas as pd
from core.history_schema import ENTITIES, HISTORY_FIELDS
from core.feature_policy import POST_EVENT_FORBIDDEN_FIELDS, validate_model_feature_list
from core.field_mapper import FieldMapper

class TemporalLeakageAuditor:
    def audit_history_features(self, result, raise_on_error=False) -> dict:
        """检查每个原始/回退来源日期严格 < target_date，禁止直接输出后验字段。"""
        features = result.features if isinstance(result.features,pd.DataFrame) else pd.DataFrame([result.features])
        metadata = result.metadata if isinstance(result.metadata,pd.DataFrame) else pd.DataFrame([result.metadata])
        failures = []
        same_day = future = invalid_source = False
        policy_ok = True
        try:
            validate_model_feature_list(list(features.columns))
        except ValueError:
            policy_ok = False
            failures.append("模型特征策略拒绝输出字段")
        if len(features) != len(metadata):
            failures.append("特征和来源行数不一致")
        canonical = {FieldMapper().canonical(name) or name for name in features.columns}
        forbidden = canonical & POST_EVENT_FORBIDDEN_FIELDS
        if forbidden:
            failures.append("出现后验原始字段：" + ",".join(sorted(forbidden)))
        if set(features) - HISTORY_FIELDS:
            failures.append("存在未经许可的历史特征字段")
        if "target_date" not in metadata or "history_max_source_date" not in metadata:
            failures.append("缺少目标日期或总体来源日期")
        else:
            targets = pd.to_datetime(metadata.target_date,errors="coerce")
            if targets.isna().any():
                failures.append("目标日期无效，无法验证时间边界")
            date_columns = [c for c in metadata if c.endswith("max_source_date")]
            for col in date_columns:
                source = pd.to_datetime(metadata[col],errors="coerce")
                same_day |= bool((source.notna() & source.eq(targets)).any())
                future |= bool((source.notna() & source.gt(targets)).any())
                invalid_source |= bool((metadata[col].notna() & source.isna()).any())
                if (source.notna() & source.ge(targets)).any():
                    failures.append(f"{col} 包含同日或未来来源")
                if (metadata[col].notna() & source.isna()).any():
                    failures.append(f"{col} 来源日期无法解析")
            if "history_visibility_mode" in metadata:
                frozen=metadata.history_visibility_mode.eq("FROZEN_CUTOFF")
                cutoffs=pd.to_datetime(metadata.get("history_cutoff_date"),errors="coerce")
                if (frozen & (cutoffs.isna() | cutoffs.ge(targets))).any():
                    failures.append("Frozen cutoff invalid or not before target")
                for col in date_columns:
                    source=pd.to_datetime(metadata[col],errors="coerce")
                    if (frozen & source.gt(cutoffs)).any():failures.append(f"{col} exceeds frozen cutoff")
            for entity in ENTITIES:
                count = features.get(f"hist_{entity}_count")
                source = metadata.get(f"hist_{entity}_max_source_date")
                if count is None or source is None:
                    failures.append(f"缺少 {entity} 数量或来源")
                elif (count.gt(0).to_numpy() & source.isna().to_numpy()).any():
                    failures.append(f"{entity} 有统计却没有来源日期")
                effective = features.get(f"hist_{entity}_delay_rate")
                effective_date = metadata.get(f"hist_{entity}_effective_max_source_date")
                if effective is not None and (effective_date is None or (effective.notna().to_numpy() & effective_date.isna().to_numpy()).any()):
                    failures.append(f"{entity} 回退值缺少可追踪来源")
        report = {"passed":not failures,"rows_checked":len(features),"rule":"all source_date < target_date; no same-day outcomes", "failures":failures,"checks":["target_date", "raw source dates", "effective fallback source dates", "same-day/future exclusion", "forbidden raw fields", "schema whitelist"]}
        dates_present = "target_date" in metadata and "history_max_source_date" in metadata
        target_ok = dates_present and pd.to_datetime(metadata.target_date, errors="coerce").notna().all()
        report["check_results"] = [
            {"name":"同日标签来源", "passed":bool(target_ok and not same_day), "description":"逐层检查原始和回退来源日期不等于目标日期"},
            {"name":"未来标签来源", "passed":bool(target_ok and not future), "description":"逐层检查原始和回退来源日期不晚于目标日期"},
            {"name":"严格日期边界", "passed":bool(target_ok and not same_day and not future and not invalid_source), "description":"所有非空 source_date < target_date"},
            {"name":"后验原始字段", "passed":not bool(forbidden), "description":"输出不含 DEP_DELAY、ARR_DELAY、实际起降时刻等原始后验字段"},
            {"name":"Feature Policy", "passed":bool(policy_ok and not (set(features)-HISTORY_FIELDS)), "description":"只接受已注册的数值历史特征；回退来源字符串仅在 metadata 中"},
            {"name":"来源可追踪性", "passed":not any("来源" in item or "行数" in item for item in failures), "description":"非空历史与有效回退值具有来源日期"},
        ]
        if failures and raise_on_error:
            raise ValueError("历史特征泄漏审计失败：" + "；".join(failures))
        return report
