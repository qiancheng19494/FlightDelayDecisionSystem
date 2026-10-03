"""构建或加载明确选择的数据集，生成真实历史验证报告。"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.bootstrap import initialize
from core.historical_features import HistoricalFeatureEngine,json_safe
from core.history_report import export_history_report,report_data

def generate(context,dataset_id,output):
    engine=HistoricalFeatureEngine(context)
    metadata=engine.build_history_cache(dataset_id)
    engine.metadata["validation"]=engine.validation_report()
    result=report_data(engine)
    export_history_report(engine,Path(output))
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset-id",type=int,required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"output/reports/stage3_history_validation.md")
    args=parser.parse_args()
    context=initialize(ROOT)
    report=generate(context,args.dataset_id,args.output)
    print(json.dumps(report["validation"]["audit"],ensure_ascii=False))
