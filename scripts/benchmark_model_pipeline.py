"""50000 行合成 DEMO，只基准 RF/ATHC 与 XGB/ATHC；临时资产不登记项目。"""
import sys,json,tempfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.create_historical_demo import create_demo
from core.historical_features import HistoricalFeatureEngine
from core.ml.training_service import TrainingService,load_model_config

def run():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);source=root/"demo_benchmark.csv";create_demo(source,50000)
        frame=pd.read_csv(source,keep_default_na=False);h=HistoricalFeatureEngine(source_path=source,cache_root=root);h.build_history_cache()
        results={}
        for model_type in ("rf","xgboost"):
            r=TrainingService(root).train_frame(frame,model_type,"COMBO_ATHC",history=h,schedule_context=frame,is_demo=True,config=load_model_config(ROOT))
            m=r["metadata"];results[model_type]={"timings":m["timings"],"save_seconds":m["save_seconds"],"load_seconds":r["load_seconds"],"total_seconds":r["total_seconds"],"model_bytes":m["model_file_size"],"training_rows":m["training_row_count"],"validation_rows":m["validation_row_count"],"serialization_equal":r["serialization_equal"]}
        report={"input_rows":len(frame),"combination":"COMBO_ATHC","raw_features":80,"data_kind":"DEMO MODEL / synthetic benchmark only","peak_memory":"未测量","models":results}
    (ROOT/"output/reports/stage5_benchmark.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":run()
