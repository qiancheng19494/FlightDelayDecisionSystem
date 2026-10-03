"""Explicit local input only; no data downloads, no final-period evaluation."""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.real_bts_validation import run
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("csv",type=Path,nargs="+");parser.add_argument("--max-rows",type=int,default=100000);parser.add_argument("--model-smoke",action="store_true");parser.add_argument("--provenance",required=True)
    args=parser.parse_args();r=run(ROOT,args.csv,args.max_rows,args.model_smoke,args.provenance,stage=lambda s:print(s,flush=True))
    print(json.dumps({k:r.get(k) for k in ["status","sample_rows","total_seconds","peak_rss_bytes","history_cache_bytes"]},ensure_ascii=False))
