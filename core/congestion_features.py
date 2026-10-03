"""C 是已知计划表代理：保留计划表所有行，不看最终取消或实际结果。"""
import numpy as np
import pandas as pd
from core.history_normalization import normalize_frame
from core.temporal_features import planned_minutes
from core.feature_groups import CONGESTION_KEYS, CONGESTION_FEATURES, CONGESTION_ALLOWED_SOURCE_FIELDS
class CongestionFeatureEngineer:
    def __init__(self):
        self.tables={};self.dates=set();self.context_rows=0;self.fitted=False
    @staticmethod
    def prepare(frame):
        f=normalize_frame(frame)
        f=f[[c for c in f if c in CONGESTION_ALLOWED_SOURCE_FIELDS]].copy()
        missing=pd.Series(np.nan,index=f.index)
        f["dep_hour"]=(planned_minutes(f.get("scheduled_departure",missing))%1440)//60
        f["arr_hour"]=(planned_minutes(f.get("scheduled_arrival",missing))%1440)//60
        return f
    def fit(self,schedule_context):return self.fit_chunks([schedule_context])
    def fit_chunks(self,chunks):
        self.tables={};self.dates=set();self.context_rows=0;self.fitted=False
        for chunk in chunks:
            f=self.prepare(chunk);self.context_rows+=len(f);self.dates.update(f.flight_date.dropna().tolist())
            for name,keys in CONGESTION_KEYS.items():
                counts=f.groupby(list(keys),dropna=True,sort=True).size().astype(float)
                self.tables[name]=self.tables[name].add(counts,fill_value=0) if name in self.tables else counts
        self.fitted=True
        return self
    def transform(self,dataframe):
        f=self.prepare(dataframe);x=pd.DataFrame(index=f.index)
        covered=f.flight_date.isin(self.dates) if self.fitted else pd.Series(False,index=f.index)
        for name,keys in CONGESTION_KEYS.items():
            valid=covered & f[list(keys)].notna().all(axis=1)
            values=np.full(len(f),np.nan)
            if valid.any() and name in self.tables:
                idx=pd.MultiIndex.from_frame(f.loc[valid,list(keys)])
                values[np.flatnonzero(valid.to_numpy())]=self.tables[name].reindex(idx,fill_value=0).to_numpy()
            x[name]=values
        x["congestion_context_available"]=covered.astype(int)
        return x[list(CONGESTION_FEATURES)]
