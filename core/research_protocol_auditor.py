"""Read-only evidence audit. Never execute research code or deserialize external pickle."""
import ast,json,hashlib
from pathlib import Path
import pandas as pd

# Reviewed implementation identities from the locked phase-5 manifest.
REVIEWED_CODE={"src/phase4_features.py":"6a2d61a32d2510c5e2c1c6f82414df7548df64cb3f567d06ff154bd2295d193c",
               "src/phase4_config.py":"4b1b87c27388e4f9bfb98f1eeca2732ca06c18bf7934c0ef9f6b301b5487beae"}
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def literal_assignment(path,name):
    tree=ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    return None

def cache_protocol(frame,cutoff="2024-11-30"):
    """Source provenance can prove rolling use; absence cannot prove a frozen policy."""
    if not {"FlightDate","Max_Source_Date"}.issubset(frame):return "UNRESOLVED"
    target=pd.to_datetime(frame.FlightDate,errors="coerce");source=pd.to_datetime(frame.Max_Source_Date,errors="coerce")
    if target.isna().any() or (source>=target).any():return "INVALID_EVIDENCE"
    validation=target.between("2024-12-01","2024-12-31")
    return "ROLLING_ASOF" if (validation & source.gt(pd.Timestamp(cutoff))).any() else "UNRESOLVED"

def parse_combinations(config_groups,result_table):
    if not isinstance(config_groups,list) or not config_groups or len(config_groups)!=len(set(config_groups)):return dict(status="UNRESOLVED",groups=[])
    if not {"Model","Feature_Group"}.issubset(result_table):return dict(status="UNRESOLVED",groups=[])
    pairs=list(result_table[["Model","Feature_Group"]].itertuples(index=False,name=None))
    expected={(m,g) for m in ("RF","XGB") for g in config_groups}
    if len(pairs)!=len(expected) or set(pairs)!=expected:return dict(status="UNRESOLVED",groups=[])
    return dict(status="CONFIRMED",groups=config_groups,model_families=["RF","XGB"],feature_combination_count=len(config_groups),experiment_count=len(expected),matches_software_16_subsets=False)

class ResearchProtocolAuditor:
    def audit(self,root):
        root=Path(root);out=dict(protocol="UNRESOLVED",validation_protocol="UNRESOLVED",training_history="UNRESOLVED",test_history="UNRESOLVED",confidence="UNRESOLVED",evidence=[],warnings=[],combinations={"status":"UNRESOLVED","groups":[]},date_split={},parameters={},paper_text_assessment="无法判断")
        def read(relative):
            path=root/relative
            if not path.is_file():out["warnings"].append("Missing: "+relative);return None
            if path.stat().st_size>4000000:out["warnings"].append("Bounded audit skipped large text: "+relative);return None
            text=path.read_text(encoding="utf-8-sig")
            out["evidence"].append(dict(path=str(path.resolve()),sha256=sha256(path),bytes=path.stat().st_size))
            return text
        try:
            code=read("src/phase4_features.py");config=read("src/phase4_config.py");train=read("src/train_phase4.py");groups_code=read("src/train_phase5.py")
            p4=read("outputs/phase4_protocol.json");p5=read("outputs/phase5/phase5_protocol.json")
            p4=json.loads(p4).get("policy",{}) if p4 else {};p5=json.loads(p5).get("protocol",{}) if p5 else {}
            out["date_split"]={k:p4.get(k) for k in ("train","validation","final_test")}
            out["parameters"]={k:p5.get(k) for k in ("rf_parameters","xgb_parameters")}
            cache_path=root/"outputs/phase5/history_leakage_samples.csv"
            if read("outputs/phase5/history_leakage_samples.csv"):
                frame=pd.read_csv(cache_path);cp=cache_protocol(frame);out["cache_validation_protocol"]=cp
                checks=[]
                for date in ["2024-12-01","2024-12-15","2024-12-31"]:
                    part=frame.loc[frame.FlightDate.eq(date)];source=pd.to_datetime(part.Max_Source_Date,errors="coerce").dropna()
                    checks.append(dict(target_date=date,rows=len(part),max_source_date=str(source.max().date()) if len(source) else None,samples=part.head(2).to_dict("records")))
                out["cache_date_checks"]=checks
                if cp=="ROLLING_ASOF":out.update(validation_protocol=cp,confidence="CONFIRMED_CACHE")
            reviewed=all((root/name).is_file() and sha256(root/name)==value for name,value in REVIEWED_CODE.items())
            manifest=p5.get("phase4_artifact_hashes",{})
            matched=all(manifest.get(name.replace("/","\\"),manifest.get(name))==value for name,value in REVIEWED_CODE.items())
            training_link=groups_code and "range(1,12)" in groups_code and "features_2024_12.pkl" in groups_code and "val.Day.min()==335 and val.Day.max()==365" in groups_code
            final_link=train and "history_indices_2024.pkl" in train and "history_values(frame,indices,day_number(c.VALID_END))" in train and "index.daily.Day.max()<=day_number(c.VALID_END)" in train
            if reviewed and matched and training_link and final_link and out.get("cache_validation_protocol")=="ROLLING_ASOF":
                out.update(protocol="MIXED",validation_protocol="ROLLING_ASOF",training_history="strict [D-7/30,D), fallback to expanding past",test_history="FROZEN_SOURCE_CUTOFF_2024-12-31; 7/30-day windows still advance with target date",confidence="CONFIRMED_CODE")
            else:out["warnings"].append("Code identity/linkage not fully confirmed; do not infer a whole-experiment protocol.")
            if groups_code and read("outputs/phase5/feature_group_ablation.csv"):
                groups=literal_assignment(root/"src/train_phase5.py","GROUPS")
                out["combinations"]=parse_combinations(groups,pd.read_csv(root/"outputs/phase5/feature_group_ablation.csv"))
            paper=read("outputs/paper/paper_v1_3_final.md")
            if paper and "验证月逐日更新历史" in paper and "时间外标签来源冻结" in paper and out["protocol"]=="MIXED":
                out["paper_text_assessment"]="已检查 paper_v1_3_final.md 对验证滚动/时间外冻结的表述与代码一致；无需修改该处。未据此代表所有 Word 版本。"
            if code:
                out["code_lines"]=[dict(line=i,text=line.strip()) for i,line in enumerate(code.splitlines(),1) if any(s in line for s in ["upper=np.minimum","range(1,13)","history_values(frame,indices","daily_aggregate(frame)"])]
            out["warnings"].extend(["论文 H 为 7/30 天窗口；软件 H 为 expanding，协议相似不代表特征或实验完全一致。","论文 16 实验=8 个含 A 的组合×2 模型，不是软件的全部 16 子集。","仅只读现存证据；未重算论文指标、未载入外部 pickle、未修改论文。"])
        except (OSError,ValueError,KeyError,SyntaxError,TypeError) as exc:
            out.update(protocol="UNRESOLVED",confidence="UNRESOLVED");out["warnings"].append(str(exc))
        return out
