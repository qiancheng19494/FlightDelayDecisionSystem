"""Read-only local inventory. No network, research execution or external pickle loading."""
import sys,json,csv,zipfile,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.research_protocol_auditor import ResearchProtocolAuditor,sha256
from core.bts_compatibility import verify
from core.ml.model_registry import atomic_json

def scan(roots,paper_root):
    files=set();errors=[]
    for root in roots:
        proc=subprocess.run(["rg","--files","--hidden","--no-ignore","-g","!**/.venv/**","-g","!**/.git/**","-g","!**/__pycache__/**","-g","!**/node_modules/**","-g","!**/_build/deps/**",str(root)],capture_output=True,text=True,encoding="utf-8",errors="replace")
        files.update(Path(p) for p in proc.stdout.splitlines());errors.extend(proc.stderr.splitlines())
    data=[];assets=[]
    for p in sorted(files):
        if p.suffix.lower() not in (".csv",".parquet",".zip",".py",".ipynb",".json",".yaml",".yml",".md",".txt",".joblib",".pkl",".npz"):continue
        record=dict(path=str(p),name=p.name,bytes=p.stat().st_size)
        if p.suffix.lower() in (".csv",".parquet",".zip") and "data" in p.parts:
            match=re.search(r"(202[45])_(\d{1,2})(?:\.|$)",p.name)
            record.update(compressed=p.suffix==".zip",possible_month=f"{match[1]}-{int(match[2]):02}" if match else None,paper_used="data/raw" in p.as_posix() and "On_Time_Reporting" in p.name,provenance="local paper download manifest" if "On_Time_Reporting" in p.name else "UNCONFIRMED_OR_DEMO")
            if p.suffix==".csv":
                try:
                    if match and match[1]=="2025":
                        with p.open(encoding="utf-8-sig",newline="") as f:record["field_count"]=len(next(csv.reader(f)))
                        record["scope"]="2025 header only; locked, no evaluation"
                    else:
                        info=verify(p,100);record.update(field_count=len(info["columns"]),sample_date_range=info.get("date_range"),mapping_valid=info["mapping"]["is_valid"],sampled_rows=info.get("sampled_rows"))
                except (OSError,ValueError) as exc:record["inspection_error"]=str(exc)
            elif p.suffix==".zip":
                try:
                    with zipfile.ZipFile(p) as z:record["members"]=[dict(name=i.filename,bytes=i.file_size) for i in z.infolist()[:30]]
                except zipfile.BadZipFile:record["inspection_error"]="Invalid zip"
            data.append(record)
        else:
            record["category"]="model_or_cache" if p.suffix in (".pkl",".joblib",".npz") else "code" if p.suffix in (".py",".ipynb") else "config_metrics_or_text"
            if p.stat().st_size<=4000000 and p.suffix not in (".pkl",".joblib",".npz"):record["sha256"]=sha256(p)
            assets.append(record)
    return dict(roots=[str(r) for r in roots],file_count=len(files),data=data,assets=assets,errors=errors,protocol_audit=ResearchProtocolAuditor().audit(paper_root),network_used=False)
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,action="append",required=True);p.add_argument("--paper-root",type=Path,required=True);args=p.parse_args()
    result=scan(args.root,args.paper_root);atomic_json(ROOT/"output/reports/stage5_2_asset_inventory.json",result)
    atomic_json(ROOT/"output/reports/stage5_2_protocol_audit.json",result["protocol_audit"])
    print(json.dumps(dict(files=result["file_count"],data_files=len(result["data"]),assets=len(result["assets"]),protocol=result["protocol_audit"]["protocol"],confidence=result["protocol_audit"]["confidence"],combinations=result["protocol_audit"]["combinations"]),ensure_ascii=False))
