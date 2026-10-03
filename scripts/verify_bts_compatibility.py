"""离线 BTS 小样本兼容性检查，不修改 CSV、不下载数据、不训练模型。"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.data_loader import FlightDataLoader
from core.history_normalization import normalize_frame
from core.time_parser import parse_hhmm,numeric
from core.delay_label import build_delay_label

from core.bts_compatibility import verify

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("csv",type=Path)
    parser.add_argument("--rows",type=int,default=100)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    try:
        result=verify(args.csv,args.rows)
        content=json.dumps(result,ensure_ascii=False,indent=2)
        print(content)
        if args.output:
            args.output.write_text(content+"\n",encoding="utf-8")
        raise SystemExit(0 if result["compatible"] else 1)
    except (OSError,ValueError) as exc:
        print(f"兼容性检查失败：{exc}",file=sys.stderr)
        raise SystemExit(1)
