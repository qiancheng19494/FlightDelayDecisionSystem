"""分块日聚合，磁盘只保存实体-日期聚合，永不写入逐条航班。"""
import csv
import json
import sqlite3
import time
from decimal import Decimal, localcontext
from pathlib import Path
from collections import Counter
import pandas as pd
from core.history_schema import ENTITIES
from core.history_normalization import normalize_frame, entity_keys
from core.delay_label import build_delay_label
from core.time_parser import numeric, parse_hhmm

def decimal_sum(values) -> str:
    with localcontext() as ctx:
        ctx.prec=400
        return str(sum((Decimal(str(v)) for v in values),Decimal(0)))

class DailyHistoryBuilder:
    """累计表保存截至来源日的和；查询用 searchsorted(side='left') 严格错开当天。"""
    def build(self, loader, directory: Path, progress=None, cancelled=None, stage=None) -> dict:
        stage = stage or (lambda text:None)
        counts = Counter()
        trends = {name:{} for name in ["month","weekday","hour"]}
        source_min = source_max = None
        aggregate_seconds = 0.0
        scan_start=time.perf_counter()
        db_path=directory/"daily_work.sqlite3"
        connection=sqlite3.connect(db_path)
        connection.create_function("decimal_add",2,lambda a,b:decimal_sum([a,b]))
        connection.execute("CREATE TABLE daily(entity TEXT, entity_key TEXT, source_date TEXT, n INTEGER, delays INTEGER, delay_sum TEXT, PRIMARY KEY(entity,entity_key,source_date)) WITHOUT ROWID")
        try:
            for raw in loader.iter_chunks(lambda p,n: progress(int(p*.65),n) if progress else None,cancelled):
                stage("正在扫描 CSV 并按日期标准化")
                frame=normalize_frame(raw)
                counts["source_rows"]+=len(frame)
                valid_dates=frame.flight_date.dropna()
                if len(valid_dates):
                    low,high=valid_dates.min().strftime("%Y-%m-%d"),valid_dates.max().strftime("%Y-%m-%d")
                    source_min=min(source_min,low) if source_min else low
                    source_max=max(source_max,high) if source_max else high
                labels=build_delay_label(frame.departure_delay,frame.get("cancelled"))
                valid=frame.flight_date.notna() & labels.notna()
                counts["invalid_date_rows"]+=int(frame.flight_date.isna().sum())
                counts["invalid_or_cancelled_label_rows"]+=int(labels.isna().sum())
                frame=frame.loc[valid].copy()
                frame["label"]=labels.loc[valid].astype(int)
                # 精确十进制求和避免 CSV 行序或分块改变浮点加法顺序。
                frame["delay"]=numeric(frame.departure_delay).astype(str)
                frame["source_date"]=frame.flight_date.dt.strftime("%Y-%m-%d")
                counts["valid_history_rows"]+=len(frame)
                counts["delay_rows"]+=int(frame.label.sum())
                aggregation_start=time.perf_counter()
                for entity,columns in ENTITIES.items():
                    if cancelled and cancelled(): raise ValueError("历史构建已取消")
                    stage("正在聚合 " + entity)
                    frame["entity_key"]=entity_keys(frame,columns)
                    grouped=frame.dropna(subset=["entity_key"]).groupby(["entity_key","source_date"],sort=False).agg(n=("label","size"),delays=("label","sum"),delay_sum=("delay",decimal_sum)).reset_index()
                    rows=[(entity,str(k),str(d),int(n),int(y),str(s)) for k,d,n,y,s in grouped.itertuples(index=False,name=None)]
                    with connection:
                        connection.executemany("INSERT INTO daily VALUES (?,?,?,?,?,?) ON CONFLICT(entity,entity_key,source_date) DO UPDATE SET n=n+excluded.n,delays=delays+excluded.delays,delay_sum=decimal_add(delay_sum,excluded.delay_sum)",rows)
                for name,keys in [("month",frame.flight_date.dt.strftime("%Y-%m")),("weekday",frame.flight_date.dt.dayofweek.astype(str))]:
                    self._trend(trends[name],keys,frame.label)
                if "scheduled_departure" in frame:
                    parsed={v:parse_hhmm(v) for v in frame.scheduled_departure.dropna().unique()}
                    hours=frame.scheduled_departure.map(lambda v: parsed[v].text[:2] if v in parsed and parsed[v].text else None)
                    self._trend(trends["hour"],hours,frame.label)
                aggregate_seconds+=time.perf_counter()-aggregation_start
            if cancelled and cancelled(): raise ValueError("历史构建已取消")
            stage("正在构建累计统计索引")
            entity_counts,aggregate_rows={},{}
            for i,entity in enumerate(ENTITIES):
                path=directory/f"{entity}_daily.csv"
                keys=set()
                row_count=0
                with path.open("w",encoding="utf-8",newline="") as stream:
                    writer=csv.writer(stream)
                    writer.writerow(["entity_key","source_date","daily_count","daily_delay_count","daily_delay_sum","cumulative_count","cumulative_delay_count","cumulative_delay_sum"])
                    last=None;count=delays=0;total=Decimal(0)
                    for key,date,n,y,s in connection.execute("SELECT entity_key,source_date,n,delays,delay_sum FROM daily WHERE entity=? ORDER BY entity_key,source_date",(entity,)):
                        if key!=last: count=delays=0;total=Decimal(0);last=key
                        count+=n;delays+=y
                        with localcontext() as ctx:
                            ctx.prec=400;total+=Decimal(s)
                        writer.writerow([key,date,n,y,s,count,delays,str(total)])
                        keys.add(key);row_count+=1
                        if row_count%10000==0 and cancelled and cancelled(): raise ValueError("历史构建已取消")
                entity_counts[entity]=len(keys);aggregate_rows[entity]=row_count
                if progress: progress(60+i*4,counts["source_rows"])
        finally:
            connection.close()
            db_path.unlink(missing_ok=True)
        return {**dict(counts),"source_min_date":source_min,"source_max_date":source_max,"entity_counts":entity_counts,"aggregate_rows":aggregate_rows,"trends":trends,"timings":{"scan_and_aggregate_seconds":round(time.perf_counter()-scan_start,4),"aggregation_seconds":round(aggregate_seconds,4)}}

    @staticmethod
    def _trend(target,keys,labels):
        frame=pd.DataFrame({"key":keys,"label":labels}).dropna(subset=["key"])
        for key,group in frame.groupby("key"):
            entry=target.setdefault(str(key),{"count":0,"delay_count":0})
            entry["count"]+=len(group);entry["delay_count"]+=int(group.label.sum())
