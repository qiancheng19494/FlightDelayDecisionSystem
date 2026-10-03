"""字段白名单 + 注册来源契约；H 的时间审计由现有引擎负责。"""
from dataclasses import asdict
import pandas as pd
from core.feature_groups import FEATURE_DEFINITIONS, FEATURE_SCHEMA_VERSION, FeatureCombinationRegistry, CONGESTION_ALLOWED_SOURCE_FIELDS
from core.feature_policy import validate_model_feature_list
class FeatureValidator:
    def validate(self,X,combination,*,lineage=None,schema_version=FEATURE_SCHEMA_VERSION,raise_on_error=True):
        failures=[];checks=[]
        c=FeatureCombinationRegistry().validate_combination(combination)
        def check(name,ok):
            checks.append({"name":name,"passed":bool(ok)})
            if not ok:failures.append(name)
        try:validate_model_feature_list(list(X));allowed=True
        except ValueError:allowed=False
        check("后验原始字段及未注册字段",allowed)
        check("无重复字段",not X.columns.duplicated().any())
        check("schema version",schema_version==FEATURE_SCHEMA_VERSION)
        check("组合字段及固定顺序",list(X)==list(c.feature_names))
        check("字段数量",len(X.columns)==len(c.feature_names))
        lineage=lineage if lineage is not None else {n:asdict(FEATURE_DEFINITIONS[n]) for n in c.feature_names}
        check("字段来源及组归属",set(lineage)==set(c.feature_names) and all(lineage.get(n)==asdict(FEATURE_DEFINITIONS[n]) for n in c.feature_names))
        dtypes_ok=True
        for name in X:
            definition=FEATURE_DEFINITIONS.get(name)
            if definition is None or isinstance(X[name],pd.DataFrame):dtypes_ok=False;continue
            if definition.dtype=="numeric" and not pd.api.types.is_numeric_dtype(X[name]):dtypes_ok=False
            if definition.dtype=="categorical" and not (pd.api.types.is_string_dtype(X[name].dtype) or isinstance(X[name].dtype,pd.CategoricalDtype)):dtypes_ok=False
        check("dtype 与 H 非字符串",dtypes_ok)
        check("C 仅计划字段 lineage",all(set(lineage.get(n,{}).get("source_fields",())).issubset(CONGESTION_ALLOWED_SOURCE_FIELDS) for n in c.feature_names if FEATURE_DEFINITIONS[n].group=="C"))
        check("标签不在 X",not any(str(n).casefold() in {"y","delay_label","label","target"} for n in X))
        report={"passed":not failures,"checks":checks,"failures":failures,"scope":"注册契约与数据类型验证；不能证明任意外部伪造值的真实来源"}
        if failures and raise_on_error:raise ValueError("特征校验失败："+"；".join(failures))
        return report
