"""HistoricalFeatureEngine：任意目标 D 仅使用 date < D 的完整自然日。

聚合表的累计值包含来源日；searchsorted(side='left') - 1 明确排除目标同日。
单条、批量、验证及 GUI 共用同一查询逻辑，不读取目标行的延误结果。
"""
import json
import logging
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
from core.data_loader import FlightDataLoader
from core.field_mapper import DatasetType
from core.history_schema import (ENTITIES, ENGINE_VERSION, SCHEMA_VERSION, HISTORY_FIELDS, history_config, HistoricalFeatureResult, HistoricalBatchResult)
from core.history_normalization import normalize_frame, entity_keys
from core.history_aggregation import DailyHistoryBuilder
from core.history_cache import HistoryCache, file_fingerprint, digest
from core.historical_fallback import HistoricalFallbackPolicy
from core.leakage_auditor import TemporalLeakageAuditor

def json_safe(value):
    """报告中以 null 表示无历史 NaN，不向数值特征中填充未来均值。"""
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,(pd.Timestamp,datetime)): return value.isoformat()
    if isinstance(value,np.generic): value=value.item()
    if value is None or value is pd.NA or value is pd.NaT: return None
    if isinstance(value,float) and not np.isfinite(value): return None
    return value

class HistoricalFeatureEngine:
    def __init__(self, context=None, *, source_path=None, cache_root=None, config=None, chunksize=100000):
        self.context=context
        self.source_path=Path(source_path).resolve() if source_path else None
        self.root=context.root if context else Path(cache_root or ".").resolve()
        self.config=history_config(config if config is not None else (context.config.values.get("historical_features") if context else None))
        self.chunksize=context.config.get("csv_chunksize") if context else chunksize
        self.dataset_id=None
        self.metadata=None
        self.tables={}
        self.index={}
        self.logger=context.logger if context else logging.getLogger("flight_delay")
        self.auditor=TemporalLeakageAuditor()

    def _select(self,dataset_id):
        if self.context:
            record=self.context.datasets.repository.get(int(dataset_id))
            if record["dataset_type"]!=DatasetType.HISTORICAL_LABELED.value:
                raise ValueError("该数据集没有真实历史标签，不能建立延误历史统计。")
            self.source_path=Path(record["file_path"]).resolve()
            self.is_demo=bool(record["is_demo"])
            self.dataset_name=record["dataset_name"]
        else:
            self.is_demo=self.source_path is not None and self.source_path.name.startswith("demo_")
            self.dataset_name=self.source_path.name if self.source_path else "未选择"
        if self.source_path is None: raise ValueError("请明确选择一个历史数据集")
        self.dataset_id=int(dataset_id)
        return HistoryCache(self.root,self.dataset_id)

    def cache_status(self,dataset_id) -> str:
        cache=self._select(dataset_id)
        _,reason=cache.load(self.source_path,self.config,verify_files=False)
        return reason

    def load_cache(self,dataset_id) -> bool:
        cache=self._select(dataset_id)
        result,reason=cache.load(self.source_path,self.config)
        if result is None:
            self.metadata=None;self.tables={};self.index={}
            self.cache_message=reason
            return False
        self.folder,self.metadata=result
        self._load_tables()
        self.cache_message=reason
        return True

    def _load_tables(self):
        self.tables={};self.index={}
        for entity in ENTITIES:
            table=pd.read_csv(self.folder/f"{entity}_daily.csv",dtype={"entity_key":"string","daily_delay_sum":"string","cumulative_delay_sum":"string"},keep_default_na=False)
            table["source_date"]=pd.to_datetime(table.source_date,format="%Y-%m-%d")
            for column in ["daily_count","daily_delay_count","cumulative_count","cumulative_delay_count"]:
                table[column]=table[column].astype("int64")
            table["cumulative_mean"]=pd.to_numeric(table.cumulative_delay_sum,errors="raise")/table.cumulative_count
            self.tables[entity]=table
            self.index[entity]={key:(group.source_date.to_numpy(dtype="datetime64[ns]"),group.cumulative_count.to_numpy(),group.cumulative_delay_count.to_numpy(),group.cumulative_mean.to_numpy(dtype=float)) for key,group in table.groupby("entity_key",sort=False)}

    def _assert_fresh(self):
        if self.metadata is None:
            raise ValueError("尚未构建或加载有效历史缓存")
        if any(self.metadata[k]!=v for k,v in file_fingerprint(self.source_path).items()):
            raise ValueError("源文件已变化，历史缓存失效，请重新构建")
        if self.metadata["configuration"]!=self.config:
            raise ValueError("历史配置已变化，请重新构建")
        if self.context and history_config(self.context.config.values.get("historical_features"))!=self.config:
            raise ValueError("当前系统历史配置已变化，请重新构建")

    def build_history_cache(self,dataset_id=0,force=False,progress=None,cancelled=None,stage=None) -> dict:
        """分块构建日累计缓存；同日无论 CSV 先后顺序都绝不参与目标 D 的查询。"""
        stage=stage or (lambda text:None)
        cache=self._select(dataset_id)
        if not force and self.load_cache(dataset_id):
            stage("已有有效历史统计缓存，直接加载")
            if progress: progress(100,self.metadata["source_rows"])
            return self.metadata
        cache.directory.mkdir(parents=True,exist_ok=True)
        generation=cache.directory/("generation_"+uuid4().hex)
        generation.mkdir()
        self.metadata=None
        start=time.perf_counter()
        self.logger.info("开始构建历史缓存 dataset=%s；严格 date < D",dataset_id)
        try:
            fingerprint=file_fingerprint(self.source_path)
            loader=FlightDataLoader(self.source_path,self.chunksize)
            loader.require_fields()
            if loader.mapping.dataset_type!=DatasetType.HISTORICAL_LABELED:
                raise ValueError("该数据集没有真实历史标签，不能建立延误历史统计。")
            warnings=[]
            if "cancelled" not in loader.mapping.mapped_fields.values():
                warnings.append("取消状态字段缺失，无法排除潜在取消航班。")
            summary=DailyHistoryBuilder().build(loader,generation,progress,cancelled,stage)
            self.folder=generation
            self.metadata={**fingerprint,**summary,"dataset_id":int(dataset_id),"dataset_name":self.dataset_name,"is_demo":self.is_demo,"engine_version":ENGINE_VERSION,"historical_feature_schema_version":SCHEMA_VERSION,"history_window_mode":"EXPANDING","min_entity_history":self.config["min_entity_history"],"configuration":self.config.copy(),"generated_at":datetime.now(timezone.utc).isoformat(),"entity_types":list(ENTITIES),"feature_schema":sorted(HISTORY_FIELDS),"warnings":warnings,"label_rule":"finite DEP_DELAY >= 15; cancelled excluded; invalid dates excluded", "time_rule":"source_date < target_date (strict natural day)","drop_exact_duplicates":False}
            if not summary.get("valid_history_rows",0): warnings.append("没有具备有效日期和标签的历史记录；所有查询返回冷启动。")
            self._load_tables()
            stage("正在验证泄漏并统计冷启动（第二遍分块）")
            validation=self.validation_report(progress=progress,cancelled=cancelled)
            self.metadata["validation"]=validation
            self.metadata["timings"]["total_build_seconds"]=round(time.perf_counter()-start,4)
            self.metadata["table_sha256"]={f"{e}_daily.csv":digest(generation/f"{e}_daily.csv") for e in ENTITIES}
            self._assert_fresh()
            if cancelled and cancelled(): raise ValueError("历史构建已取消，未发布新缓存")
            stage("正在写入历史缓存和审计结果")
            cache.publish(generation,self.metadata)
            self.logger.info("历史缓存完成 dataset=%s；有效历史=%s；审计=%s",dataset_id,summary.get("valid_history_rows",0),validation["audit"]["passed"])
            if progress: progress(100,self.metadata["source_rows"])
            return self.metadata
        except Exception:
            self.metadata=None;self.tables={};self.index={}
            # 只清理本次创建的已解析缓存子目录，保留旧版本与源文件。
            if generation.resolve().parent==cache.directory.resolve():
                shutil.rmtree(generation)
            self.logger.exception("历史缓存构建失败 dataset=%s",dataset_id)
            raise

    def transform_chunk(self,df: pd.DataFrame, *, audit=None) -> HistoricalBatchResult:
        """只提取日期和实体键；目标 DEP_DELAY、ARR_DELAY 等后验值一律忽略。

        合法目标日期不足时拒绝该批次，调用方须显式处理无效日期，不能默默填 0。
        """
        self._assert_fresh()
        original_index=df.index.copy()
        frame=normalize_frame(df).reset_index(drop=True)
        if frame.flight_date.isna().any():
            raise ValueError("目标日期无效，不能构造可审计的历史特征")
        dates=frame.flight_date.to_numpy(dtype="datetime64[ns]")
        features=pd.DataFrame(index=frame.index)
        metadata=pd.DataFrame({"target_date":frame.flight_date})
        threshold=self.config["min_entity_history"]
        for entity,columns in ENTITIES.items():
            keys=entity_keys(frame,columns)
            n=len(frame)
            counts=np.zeros(n,dtype="int64");delays=np.zeros(n,dtype="int64")
            means=np.full(n,np.nan)
            sources=np.full(n,np.datetime64("NaT"),dtype="datetime64[ns]")
            for key,indices in keys.dropna().groupby(keys.dropna(),sort=False).groups.items():
                history=self.index[entity].get(key)
                if history is None: continue
                positions=np.asarray(list(indices),dtype=int)
                # 左侧插入点之前一项：严格排除同日，缺失自然日仍获得最新过去累计值。
                offsets=np.searchsorted(history[0],dates[positions],side="left")-1
                good=offsets>=0
                target=positions[good];offsets=offsets[good]
                sources[target]=history[0][offsets];counts[target]=history[1][offsets]
                delays[target]=history[2][offsets];means[target]=history[3][offsets]
            prefix=f"hist_{entity}"
            features[prefix+"_count"]=counts
            features[prefix+"_delay_count"]=delays
            features[prefix+"_delay_rate_raw"]=np.divide(delays,counts,out=np.full(n,np.nan),where=counts>0)
            features[prefix+"_mean_dep_delay_raw"]=means
            features[prefix+"_available"]=(counts>=threshold).astype(int)
            features[f"cold_start_{entity}"]=(counts<threshold).astype(int)
            metadata[prefix+"_max_source_date"]=sources
            metadata[prefix+"_status"]=np.where(counts==0,"NO_HISTORY",np.where(counts<threshold,"INSUFFICIENT_HISTORY","AVAILABLE"))
        HistoricalFallbackPolicy().apply(features,metadata)
        result=HistoricalBatchResult(features,metadata,list(self.metadata["warnings"]),{})
        if self.config["enable_leakage_audit"] if audit is None else audit:
            self.auditor.audit_history_features(result,raise_on_error=True)
        features.index=original_index;metadata.index=original_index
        return result

    def transform_single(self,record: dict,as_of_date=None) -> HistoricalFeatureResult:
        """与批量共用 strict-as-of；不能用晚于目标航班日期的 as_of_date 查询。"""
        normalized=record.copy()
        from core.field_mapper import FieldMapper
        normalized={FieldMapper().canonical(k) or k:v for k,v in normalized.items()}
        if as_of_date is not None:
            requested=pd.Timestamp(as_of_date).normalize()
            if "flight_date" in normalized and requested>pd.Timestamp(normalized["flight_date"]).normalize():
                raise ValueError("as_of_date 不能晚于目标航班日期")
            normalized["flight_date"]=requested.strftime("%Y-%m-%d")
        result=self.transform_chunk(pd.DataFrame([normalized]))
        features={name:result.features[name].iloc[0].item() if isinstance(result.features[name].iloc[0],np.generic) else result.features[name].iloc[0] for name in result.features}
        metadata=result.metadata.iloc[0].to_dict()
        cold={e:{"cold_start":int(features[f"cold_start_{e}"]),"status":metadata[f"hist_{e}_status"],"fallback_source":metadata[f"hist_{e}_fallback_source"]} for e in ENTITIES}
        return HistoricalFeatureResult(features,metadata,result.warnings,cold)

    def transform_dataset(self,path=None,progress=None,cancelled=None):
        """迭代返回批次结果，不一次收集全量特征；无效目标日期显式抛错。"""
        loader=FlightDataLoader(path or self.source_path,self.chunksize)
        for chunk in loader.iter_chunks(progress,cancelled):
            yield self.transform_chunk(chunk)

    def validation_report(self,progress=None,cancelled=None) -> dict:
        """分块审计全部有效日期目标，并统计真实冷启动/回退次数。"""
        self._assert_fresh()
        cold=Counter({e:0 for e in ENTITIES});fallback={e:Counter() for e in ENTITIES}
        targets=invalid=first_count=checked=0
        samples=[]
        audits=[]
        first_day_valid=True
        mode="FULL" if self.config["enable_leakage_audit"] else "SAMPLED_PER_CHUNK"
        loader=FlightDataLoader(self.source_path,self.chunksize)
        for chunk in loader.iter_chunks(cancelled=cancelled):
            normalized=normalize_frame(chunk)
            valid=normalized.flight_date.notna()
            invalid+=int((~valid).sum())
            # 转回日字符串以保持与导入接口一致；后验列不影响特征。
            targets_frame=normalized.loc[valid,["flight_date","airline","origin","destination"]].copy()
            targets_frame["flight_date"]=targets_frame.flight_date.dt.strftime("%Y-%m-%d")
            batch=self.transform_chunk(targets_frame,audit=False)
            n=len(batch.features);targets+=n
            if n:
                audit_batch=batch
                if mode!="FULL":
                    positions=sorted(set(range(min(n,50)))|set(range(max(0,n-50),n)))
                    audit_batch=HistoricalBatchResult(batch.features.iloc[positions],batch.metadata.iloc[positions])
                audit=self.auditor.audit_history_features(audit_batch,raise_on_error=True)
                audits.append(audit)
                checked+=audit["rows_checked"]
                for e in ENTITIES:
                    cold[e]+=int(batch.features[f"cold_start_{e}"].sum())
                    fallback[e].update(batch.metadata[f"hist_{e}_fallback_source"].value_counts().to_dict())
                first=batch.metadata.target_date.eq(pd.Timestamp(self.metadata["source_min_date"]))
                first_count+=int(first.sum())
                first_day_valid &= bool(batch.features.loc[first,"hist_global_count"].eq(0).all())
                if not first_day_valid:
                    raise ValueError("第一日期错误使用了未来全局统计")
                for index in range(min(n,3)):
                    if len(samples)<6:
                        samples.append(json_safe({"target":targets_frame.iloc[index].to_dict(),"features":batch.features.iloc[index].to_dict(),"metadata":batch.metadata.iloc[index].to_dict()}))
            if progress: progress(min(97,85+int(targets/max(self.metadata["source_rows"],1)*12)),targets+invalid)
        self._assert_fresh()
        schema_ok = self.metadata["historical_feature_schema_version"] == SCHEMA_VERSION and self.metadata["engine_version"] == ENGINE_VERSION
        checks=[]
        if audits:
            for item in audits[0]["check_results"]:
                checks.append({**item,"passed":all(next(x["passed"] for x in a["check_results"] if x["name"]==item["name"]) for a in audits)})
        checks += [{"name":"第一历史日", "passed":first_day_valid,"description":f"检查 {first_count} 条首日记录的 global_count 为 0"}, {"name":"缓存 schema / engine", "passed":schema_ok,"description":"运行时 metadata 与当前特征 schema、引擎版本一致"}]
        audit_result={"passed":all(a["passed"] for a in audits) and all(c["passed"] for c in checks),"rows_checked":checked,"mode":mode,"rule":"all raw and fallback source dates < target_date","first_date_global_count_zero":first_day_valid,"check_results":checks,"coverage_note":"无有效目标时仅验证结构" if not checked else "检查范围见 mode 和 rows_checked"}
        return {"dataset_id":self.dataset_id,"dataset_source":str(self.source_path),"time_range":[self.metadata["source_min_date"],self.metadata["source_max_date"]],"feature_schema":sorted(HISTORY_FIELDS),"entity_counts":self.metadata["entity_counts"],"target_rows":targets,"invalid_target_dates_skipped":invalid,"cold_start_counts":dict(cold),"fallback_source_frequency":{e:dict(c) for e,c in fallback.items()},"first_date_without_history":first_count,"audit":audit_result,"sample_queries":samples}

    def entity_summary(self,entity: str,min_count=1,sort_by="count") -> pd.DataFrame:
        """整个选定历史数据集的描述统计，不作为目标 D 的历史特征。"""
        self._assert_fresh()
        if entity not in ENTITIES: raise ValueError("未知实体类型")
        table=self.tables[entity].groupby("entity_key",sort=False).tail(1).copy()
        result=pd.DataFrame({"entity":table.entity_key.map(lambda k:" → ".join(json.loads(k)) or "GLOBAL"),"entity_key":table.entity_key,"history_count":table.cumulative_count,"history_delay_count":table.cumulative_delay_count,"delay_rate":table.cumulative_delay_count/table.cumulative_count,"mean_dep_delay":table.cumulative_mean})
        column="delay_rate" if sort_by=="delay_rate" else "history_count"
        return result.loc[result.history_count>=min_count].sort_values(column,ascending=False).reset_index(drop=True)

    def entity_trend(self,entity: str,key: str) -> pd.DataFrame:
        self._assert_fresh()
        table=self.tables[entity]
        result=table.loc[table.entity_key.eq(key),["source_date","daily_count","daily_delay_count"]].copy()
        result["delay_rate"]=result.daily_delay_count/result.daily_count
        return result
