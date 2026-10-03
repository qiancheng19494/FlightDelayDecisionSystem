"""十万行确定性计划 DEMO；历史缓存构建与 Pipeline 计时分开。"""
import json,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from core.historical_features import HistoricalFeatureEngine
from core.congestion_features import CongestionFeatureEngineer
from core.feature_pipeline import FeaturePipeline
from core.temporal_features import TemporalFeatureEngineer
from scripts.create_historical_demo import create_demo

def benchmark(rows=100000):
    with tempfile.TemporaryDirectory() as directory:
        source=Path(directory)/"demo.csv";create_demo(source,rows)
        frame=pd.read_csv(source,keep_default_na=False)
        # 显式合成计划到达/时长，保留真实 DEMO 标识；不是 BTS 实验数据。
        frame["CRS_ARR_TIME"]=1200;frame["CRS_ELAPSED_TIME"]=120;frame["DISTANCE"]=500
        frame.to_csv(source,index=False)
        history=HistoricalFeatureEngine(source_path=source,cache_root=directory);history.build_history_cache()
        start=time.perf_counter();TemporalFeatureEngineer().transform(frame);t=time.perf_counter()-start
        start=time.perf_counter();c=CongestionFeatureEngineer().fit(frame);aggregate=time.perf_counter()-start
        start=time.perf_counter();c.transform(frame);lookup=time.perf_counter()-start
        start=time.perf_counter();history.transform_chunk(frame,audit=True);h=time.perf_counter()-start
        result=FeaturePipeline(history).build_features(frame,mode="TRAIN",schedule_context=c)
        return {"rows":rows,"columns":len(result.X.columns),"T_seconds":round(t,4),"C_aggregation_seconds":round(aggregate,4),"C_lookup_seconds":round(lookup,4),"H_query_and_audit_seconds":round(h,4),"pipeline_timings":result.metadata["timings"],"total_including_C_aggregation_seconds":round(result.metadata["timings"]["total_seconds"]+aggregate,4),"matrix_memory_bytes":int(result.X.memory_usage(index=True,deep=True).sum()),"peak_process_memory":"未测量；matrix_memory_bytes 仅为输出 DataFrame 占用","history_audit":result.metadata["history_audit"],"feature_audit":result.metadata["validation"],"data_kind":"synthetic DEMO; not paper experiment"}
if __name__=="__main__":
    r=benchmark();(ROOT/"output/reports/stage4_benchmark.json").write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(r,ensure_ascii=False))
