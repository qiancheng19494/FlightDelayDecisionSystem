"""Bounded offline real-format pipeline smoke, isolated mutations and explicit model identity."""
import ctypes,json,time,tempfile,uuid,threading
from pathlib import Path
import numpy as np
import pandas as pd
from core.data_loader import FlightDataLoader
from core.bts_compatibility import quality_frame,verify
from core.history_normalization import normalize_frame
from core.historical_features import HistoricalFeatureEngine,json_safe
from core.feature_pipeline import FeaturePipeline
from core.feature_service import audit_result
from core.congestion_features import CongestionFeatureEngineer
from core.ml.time_split import TemporalDataSplitter,LOCK_DATE
from core.ml.training_service import TrainingService
from core.ml.model_registry import atomic_json
from core.history_cache import file_fingerprint,digest
from core.bootstrap import initialize

class MemoryMonitor:
    """Windows process peak working set (OS high-water mark), or sampled psutil RSS."""
    def __init__(self):self.peak=None;self.method="unavailable";self.stop=threading.Event()
    def sample(self):
        if __import__("os").name=="nt":
            class Counters(ctypes.Structure):
                _fields_=[("cb",ctypes.c_ulong),("PageFaultCount",ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ["PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage"]]
            c=Counters();c.cb=ctypes.sizeof(c)
            ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
            handle=ctypes.windll.kernel32.GetCurrentProcess()
            fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
            if fn(handle,ctypes.byref(c),c.cb):self.method="Windows PeakWorkingSetSize; whole-process high-water mark";return int(c.PeakWorkingSetSize)
        try:
            import psutil
            self.method="psutil sampled RSS every 50 ms";return psutil.Process().memory_info().rss
        except ImportError:return None
    def __enter__(self):
        def run():
            while not self.stop.is_set():
                value=self.sample()
                if value is not None:self.peak=max(self.peak or 0,value)
                self.stop.wait(.05)
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join();v=self.sample();self.peak=max(self.peak or 0,v or 0) or None

def read_bounded(path,max_rows):
    if not 1<=max_rows<=100000:raise ValueError("安全上限 100000 行；不自动放大全量")
    loader=FlightDataLoader(path,chunksize=min(max_rows,5000));loader.require_fields();parts=[];n=0
    for chunk in loader.iter_chunks():
        part=chunk.iloc[:max_rows-n].copy();dates=normalize_frame(part).flight_date
        if dates.ge(LOCK_DATE).any():raise ValueError("最终测试日期锁定，不接收 2025-01 及以后样本")
        parts.append(part);n+=len(part)
        if n>=max_rows:break
    if not parts:raise ValueError("数据为空")
    return pd.concat(parts,ignore_index=True)

def check_AT(frame):
    sample=normalize_frame(frame).head(100);r=FeaturePipeline().build_features(sample,"COMBO_AT")
    def minute(v):
        try:
            n=float(v)
            if not np.isfinite(n) or n!=int(n):return np.nan
            n=int(n)
            if n==2400:return 1440.
            h,m=divmod(n,100)
            return float(h*60+m) if 0<=h<24 and 0<=m<60 else np.nan
        except (TypeError,ValueError):return np.nan
    for name,hour in [("scheduled_departure","dep_hour"),("scheduled_arrival","arr_hour")]:
        expected=sample[name].map(minute).to_numpy();np.testing.assert_allclose(r.X[name],expected,equal_nan=True)
        np.testing.assert_allclose(r.X[hour],expected%1440//60,equal_nan=True)
    expected=sample.flight_date.dt.dayofweek.to_numpy();np.testing.assert_allclose(r.X.day_of_week,expected)
    np.testing.assert_allclose(r.X.month,sample.flight_date.dt.month)
    np.testing.assert_allclose(r.X.is_weekend,(expected>=5).astype(float))
    for name,period,offset in [("dep_hour",24,0),("day_of_week",7,0),("month",12,1)]:
        angle=2*np.pi*(r.X[name]-offset)/period
        np.testing.assert_allclose(r.X[name+"_sin"],np.sin(angle),equal_nan=True);np.testing.assert_allclose(r.X[name+"_cos"],np.cos(angle),equal_nan=True)
    expected=["NIGHT" if h<6 else "MORNING" if h<12 else "AFTERNOON" if h<18 else "EVENING" if pd.notna(h) else None for h in r.X.dep_hour]
    assert r.X.time_of_day.fillna("MISSING").tolist()==[v or "MISSING" for v in expected]
    return dict(passed=True,rows=len(sample),fields="planned minutes, hours, weekday, month, weekend, time_of_day, cyclic encodings")

def validate_frame(frame,history,work,stage=lambda text:None,date_ranges=None):
    dates=normalize_frame(frame).flight_date
    if dates.isna().any():raise ValueError("无效日期：质量报告保留异常，H smoke 必须显式先处理")
    if dates.ge(LOCK_DATE).any():raise ValueError("最终测试日期锁定")
    split=TemporalDataSplitter().split(dates,real_smoke=True,**(date_ranges or {}))
    cutoff=split.metadata["training_end_date"];train=frame.iloc[split.train_positions];validation=frame.iloc[split.validation_positions]
    stage("构造 A/T/H/C 与实际来源审计")
    schedule=CongestionFeatureEngineer().fit(frame);pipeline=FeaturePipeline(history)
    a=pipeline.build_features(frame,"COMBO_A",mode="TRAIN")
    tr=pipeline.build_features(train,"COMBO_ATHC",mode="TRAIN",schedule_context=schedule)
    frozen=pipeline.build_features(validation,"COMBO_ATHC",mode="VALIDATION",schedule_context=schedule,history_cutoff_date=cutoff)
    rolling=pipeline.build_features(validation,"COMBO_ATHC",mode="VALIDATION",schedule_context=schedule,history_visibility_mode="ROLLING_ASOF")
    audits={name:audit_result(r) for name,r in [("A",a),("train",tr),("frozen",frozen),("rolling",rolling)]}
    assert all(r["passed"] for r in audits.values())
    stage("真实样本：修改验证标签后的 Frozen / Rolling 变形验证")
    from core.feature_groups import HISTORY_FEATURES
    source=normalize_frame(frame);changed=source.copy();eligible=changed.flight_date.gt(pd.Timestamp(cutoff));changed.loc[eligible,"departure_delay"]="9999"
    mutated=Path(work)/"mutated_history.csv";changed.to_csv(mutated,index=False)
    h2=HistoricalFeatureEngine(source_path=mutated,cache_root=Path(work)/"mutated_cache");h2.build_history_cache(1)
    f2=FeaturePipeline(h2).build_features(validation,"COMBO_H",mode="VALIDATION",history_cutoff_date=cutoff)
    pd.testing.assert_frame_equal(frozen.X[list(HISTORY_FEATURES)],f2.X,check_exact=True)
    r2=FeaturePipeline(h2).build_features(validation,"COMBO_H",history_visibility_mode="ROLLING_ASOF")
    vd=normalize_frame(validation).flight_date;first=vd.eq(vd.min());later=~first
    pd.testing.assert_frame_equal(rolling.X.loc[first,list(HISTORY_FEATURES)],r2.X.loc[first],check_exact=True)
    rolling_changed=bool((rolling.X.loc[later,"hist_global_mean_dep_delay_raw"]!=r2.X.loc[later,"hist_global_mean_dep_delay_raw"]).any()) if later.any() else False
    assert rolling_changed,"Need later validation date with observed labels to demonstrate rolling"
    stage("验证计划拥堵特征不受后验字段影响")
    altered=source.copy()
    for col in ("departure_delay","arrival_delay","cancelled","actual_departure_time","actual_arrival_time"):altered[col]=999
    c1=pipeline.build_features(frame.head(100),"COMBO_C",schedule_context=schedule)
    c2=pipeline.build_features(altered.head(100),"COMBO_C",schedule_context=altered)
    pd.testing.assert_frame_equal(c1.X,c2.X,check_exact=True)
    probes=[]
    for r,name in [(tr,"TRAIN"),(frozen,"FROZEN"),(rolling,"ROLLING")]:
        hm=r.history_metadata
        for date,part in list(hm.groupby("target_date"))[:3]:
            probes.append(dict(mode=name,target_date=str(date.date()),max_source_date=str(pd.to_datetime(part.history_max_source_date).max().date()) if part.history_max_source_date.notna().any() else None))
    return dict(passed=True,split=split.metadata,cutoff=cutoff,audits=audits,A_T=check_AT(frame),frozen_invariance=True,rolling_same_day_invariance=True,rolling_later_change=rolling_changed,C_posterior_independence=True,
                history_probes=probes,matrix_rows=len(frame),matrix_columns=len(frozen.X.columns),matrix_deep_bytes=int(tr.X.memory_usage(deep=True).sum()+frozen.X.memory_usage(deep=True).sum()),schedule_scope="Explicit sampled schedule only; not a complete airport/month plan or historical real-time snapshot")

def run(root,paths,max_rows=100000,model_smoke=False,provenance=None,stage=print):
    root=Path(root);report_dir=root/"output/reports";report_dir.mkdir(parents=True,exist_ok=True)
    if not paths:return dict(status="NOT EXECUTED - DATA NOT FOUND",models=[])
    if not 2<=max_rows<=100000:raise ValueError("本阶段上限100000行")
    if provenance is None:raise ValueError("必须记录真实数据来源依据；不能从文件名或 schema 猜测 REAL 身份")
    if any("demo" in str(p).lower() or "fixtures" in Path(p).parts for p in paths):raise ValueError("DEMO/fixture 不能标 REAL_SMOKE")
    started=time.perf_counter();report=dict(status="STARTED",provenance=provenance,source_fingerprints={str(Path(p).resolve()):file_fingerprint(Path(p)) for p in paths},max_rows=max_rows,models=[],final_test_evaluated=False)
    context=None
    with MemoryMonitor() as memory:
        try:
            stage("只读有界采样；完整原始文件不导入、不改写")
            report["headers"]=[verify(p,100) for p in paths]
            parts=[read_bounded(p,max_rows//len(paths)) for p in paths]
            date_ranges=None
            if len(parts)==2:
                d1=normalize_frame(parts[0]).flight_date;d2=normalize_frame(parts[1]).flight_date
                if d1.max()>=d2.min():raise ValueError("双源 smoke 需训练源全部日期早于验证源")
                date_ranges=dict(train_start=str(d1.min().date()),train_end=str(d1.max().date()),validation_start=str(d2.min().date()),validation_end=str(d2.max().date()))
            raw=pd.concat(parts,ignore_index=True);del parts
            report["quality"]=quality_frame(raw);atomic_json(report_dir/"stage5_2_real_bts_data_quality.json",json_safe(report["quality"]))
            if report["quality"]["missing_bts_fields"]:raise ValueError("缺少 BTS smoke 字段")
            frame=normalize_frame(raw);keep=[c for c in frame if c in __import__("core.field_mapper",fromlist=["ALIASES"]).ALIASES]
            frame=frame[keep].copy();del raw
            sample_path=root/"data/raw"/("real_bts_smoke_"+uuid.uuid4().hex+".csv");sample_path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(sample_path,index=False)
            report.update(sample_path=str(sample_path),sample_sha256=digest(sample_path),sample_rows=len(frame),sample_method="first bounded records per source, not representative; only selected rows form H and C context")
            context=initialize(root);record=context.datasets.import_dataset(sample_path);report["dataset_id"]=record["id"]
            assert not record["is_demo"]
            history=HistoricalFeatureEngine(context);stage("构建独立真实小样本历史缓存")
            tick=time.perf_counter();history.build_history_cache(record["id"]);report["history_build_seconds"]=time.perf_counter()-tick
            report["history_cache_bytes"]=sum(p.stat().st_size for p in history.folder.rglob("*") if p.is_file())
            with tempfile.TemporaryDirectory(prefix="bts_invariance_") as work:report["features"]=validate_frame(frame,history,work,stage,date_ranges)
            if model_smoke:
                source_fingerprints={**report["source_fingerprints"],str(sample_path):file_fingerprint(sample_path)}
                for kind in ("rf","xgboost"):
                    stage("训练 REAL_SMOKE ATHC "+kind)
                    result=TrainingService(root).train_frame(frame,kind,"COMBO_ATHC",history=history,schedule_context=frame,is_demo=False,formal_requested=True,dataset_id=record["id"],fingerprints=source_fingerprints,training_purpose="REAL_SMOKE",stage=stage,date_ranges=date_ranges)
                    report["models"].append(dict(metadata=result["metadata"],serialization_equal=result["serialization_equal"],total_seconds=result["total_seconds"],model_path=result["model_path"]))
            for path,fp in report["source_fingerprints"].items():
                if file_fingerprint(Path(path))!=fp:raise ValueError("运行期间原始文件变化")
            report["status"]="PASS"
        except Exception as exc:
            report.update(status="FAIL",error=str(exc));raise
        finally:
            if context:
                for handler in list(context.logger.handlers):handler.close();context.logger.removeHandler(handler)
            report["total_seconds"]=time.perf_counter()-started
            report["peak_rss_bytes"]=memory.sample();report["memory_method"]=memory.method
            atomic_json(report_dir/"stage5_2_real_bts_validation.json",json_safe(report))
    return report
