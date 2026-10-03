"""显式日期窗口或 DEMO 唯一日期边界；2025-01 及之后锁定。"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from core.time_parser import parse_dates
LOCK_DATE=pd.Timestamp("2025-01-01")
@dataclass
class TemporalSplit:
    train_positions: np.ndarray
    validation_positions: np.ndarray
    metadata: dict
class TemporalDataSplitter:
    def split(self,dates,*,is_demo=False,real_smoke=False,train_start=None,train_end=None,validation_start=None,validation_end=None):
        d=pd.Series(dates).reset_index(drop=True)
        d=d.dt.normalize().dt.tz_localize(None) if pd.api.types.is_datetime64_any_dtype(d) else parse_dates(d)
        if d.isna().any():raise ValueError("日期无效，禁止时间划分")
        if d.ge(LOCK_DATE).any():raise ValueError("最终时间外测试集已锁定：Stage 5 不接收 2025-01 及之后记录")
        args=(train_start,train_end,validation_start,validation_end)
        if any(v is not None for v in args) and not all(v is not None for v in args):raise ValueError("请提供完整训练及验证日期区间")
        if all(v is not None for v in args):a,b,c,e=map(pd.Timestamp,args)
        elif is_demo or real_smoke:
            unique=sorted(d.unique())
            if len(unique)<2:raise ValueError("至少需要两个不同日期")
            boundary=max(1,min(len(unique)-1,int(len(unique)*.8)))
            a,b,c,e=map(pd.Timestamp,[unique[0],unique[boundary-1],unique[boundary],unique[-1]])
        else:a,b,c,e=map(pd.Timestamp,["2024-01-01","2024-11-30","2024-12-01","2024-12-31"])
        if not (a<=b<c<=e<LOCK_DATE):raise ValueError("训练/验证区间重叠、反序或触及锁定集")
        if not is_demo and not real_smoke and (a,b,c,e)!=tuple(map(pd.Timestamp,["2024-01-01","2024-11-30","2024-12-01","2024-12-31"])):raise ValueError("正式数据必须使用既定 2024 年训练/验证窗口")
        train=np.flatnonzero(d.between(a,b));valid=np.flatnonzero(d.between(c,e))
        if not len(train) or not len(valid):raise ValueError("训练集或验证集为空")
        return TemporalSplit(train,valid,{"training_start_date":str(a.date()),"training_end_date":str(b.date()),"validation_start_date":str(c.date()),"validation_end_date":str(e.date()),"train_rows_before_label_filter":len(train),"validation_rows_before_label_filter":len(valid),"excluded_rows":len(d)-len(train)-len(valid),"train_max_date":str(d.iloc[train].max().date()),"validation_min_date":str(d.iloc[valid].min().date()),"final_test_locked":True,"shuffle":False})
