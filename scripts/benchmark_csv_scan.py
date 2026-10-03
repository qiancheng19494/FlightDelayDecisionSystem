"""可选十万行扫描基准；仅生成有限大小 DEMO 测试数据。"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.create_demo_data import create_demo
from core.data_loader import FlightDataLoader

def benchmark(path=None, rows=100000, chunksize=10000):
    with tempfile.TemporaryDirectory(prefix="flight_benchmark_") as directory:
        source=Path(path) if path else Path(directory)/"demo_benchmark.csv"
        if path is None:
            create_demo(source,rows,anomalies=False)
        start=time.perf_counter()
        progress=[]
        report=FlightDataLoader(source,chunksize).scan_metadata(lambda value,count: progress.append((value,count)),cache_directory=directory)
        elapsed=time.perf_counter()-start
        result={"rows":report["row_count"],"file_bytes":source.stat().st_size,"chunksize":chunksize,"seconds":round(elapsed,3),"rows_per_second":round(report["row_count"]/elapsed),"progress_callbacks":len(progress),"duplicate_rows":report["duplicate_rows"],"valid_labeled":report["valid_labeled_count"],"demo_only":path is None}
        if path is None:
            assert report["row_count"] == rows
        return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--rows",type=int,default=100000)
    parser.add_argument("--chunksize",type=int,default=10000)
    parser.add_argument("--output")
    args=parser.parse_args()
    if not 1<=args.rows<=1000000:
        parser.error("生成记录数应在 1 至 1000000 之间")
    result=benchmark(args.file,args.rows,args.chunksize)
    content=json.dumps(result,ensure_ascii=False,indent=2)
    print(content)
    if args.output:
        Path(args.output).write_text(content+"\n",encoding="utf-8")
