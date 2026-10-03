"""复用 BTS 解析；缺失不以午夜冒充；星期从周一 0 开始。"""
import numpy as np
import pandas as pd
from core.time_parser import parse_hhmm
from core.history_normalization import normalize_frame
from core.feature_groups import TEMPORAL_FEATURES
TIME_OF_DAY_BOUNDARIES=((0,6,"NIGHT"),(6,12,"MORNING"),(12,18,"AFTERNOON"),(18,24,"EVENING"))
def planned_minutes(values):
    def minutes(value):
        parsed=parse_hhmm(value)
        if parsed.text is None:return np.nan
        h,m=map(int,parsed.text.split(":"))
        return h*60+m+1440*parsed.day_offset
    return values.map(minutes).astype(float)
class TemporalFeatureEngineer:
    def transform(self,dataframe):
        f=normalize_frame(dataframe);x=pd.DataFrame(index=f.index)
        missing=pd.Series(np.nan,index=f.index)
        dep=planned_minutes(f.get("scheduled_departure",missing));arr=planned_minutes(f.get("scheduled_arrival",missing))
        x["dep_hour"]=(dep%1440)//60;x["arr_hour"]=(arr%1440)//60
        for name,value in [("day_of_week",f.flight_date.dt.dayofweek),("month",f.flight_date.dt.month),("day_of_month",f.flight_date.dt.day)]:x[name]=value.astype(float)
        x["is_weekend"]=x.day_of_week.ge(5).astype(float).where(f.flight_date.notna())
        x["time_of_day"]=pd.Series(pd.NA,index=f.index,dtype="string")
        for lower,upper,label in TIME_OF_DAY_BOUNDARIES:x.loc[x.dep_hour.ge(lower)&x.dep_hour.lt(upper),"time_of_day"]=label
        for name,period,offset in [("dep_hour",24,0),("day_of_week",7,0),("month",12,1)]:
            angle=2*np.pi*(x[name]-offset)/period
            x[name+"_sin"]=np.sin(angle);x[name+"_cos"]=np.cos(angle)
        x["dep_time_missing"]=dep.isna().astype(int);x["arr_time_missing"]=arr.isna().astype(int);x["flight_date_missing"]=f.flight_date.isna().astype(int)
        return x[list(TEMPORAL_FEATURES)]
