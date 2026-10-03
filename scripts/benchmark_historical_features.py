"""可选十万行历史基准；仅生成临时 DEMO，不导入正式数据库。"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.create_historical_demo import create_demo
from core.historical_features import HistoricalFeatureEngine

def run(rows=100000):
    with tempfile.TemporaryDirectory(prefix="history_benchmark_") as directory:
        root=Path(directory);source=root/"demo_benchmark_history.csv"
        create_demo(source,rows)
        engine=HistoricalFeatureEngine(source_path=source,cache_root=root,chunksize=10000)
        start=time.perf_counter();metadata=engine.build_history_cache();elapsed=time.perf_counter()-start
        start=time.perf_counter()
        for _ in range(100):
            engine.transform_single(dict(flight_date="2024-04-01",airline="AA",origin="AAA",destination="BBB"))
        latency=(time.perf_counter()-start)/100
        return {"rows":rows,"timings":metadata["timings"],"wall_seconds":round(elapsed,4),"cache_bytes":sum(p.stat().st_size for p in engine.folder.iterdir()),"aggregate_rows":metadata["aggregate_rows"],"mean_single_query_ms":round(latency*1000,3),"audit":metadata["validation"]["audit"],"peak_memory":"未测量；磁盘中间聚合，内存加载日级缓存索引，未载入原始全表"}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--rows",type=int,default=100000)
    parser.add_argument("--output",type=Path,default=ROOT/"output/reports/stage3_benchmark.json")
    args=parser.parse_args()
    if not 1<=args.rows<=500000: parser.error("测试行数须在 1–500000")
    report=run(args.rows)
    text=json.dumps(report,ensure_ascii=False,indent=2)
    args.output.write_text(text,encoding="utf-8")
    print(text)
