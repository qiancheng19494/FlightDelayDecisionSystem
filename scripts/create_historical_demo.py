"""90 天确定性 DEMO，包含中途出现的实体和冷启动，不覆盖第二阶段 DEMO。"""
import csv
from datetime import date,timedelta
from pathlib import Path

def create_demo(path: Path,rows=540):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["FL_DATE","OP_UNIQUE_CARRIER","ORIGIN","DEST","CRS_DEP_TIME","DEP_DELAY","CANCELLED","OP_CARRIER_FL_NUM","DEMO_NOTICE"])
        for i in range(rows):
            day=(i//6)%90
            airline,origin,dest=[("AA","AAA","BBB"),("AA","AAA","BBB"),("BB","CCC","DDD"),("BB","DDD","CCC"),("AA","BBB","AAA"),("BB","CCC","BBB")][i%6]
            if day>=30 and i%6==5: airline="ZX"
            if day>=45 and i%6==4: origin="NEW"
            if day>=60 and i%6==3: origin,dest="AAA","DDD"
            delay=[0,20,14.9,15,15.1,45][(i+day)%6]
            writer.writerow([(date(2024,1,1)+timedelta(days=day)).isoformat(),airline,origin,dest,[5,730,930,1530,1830,2400][i%6],"" if i%113==112 else delay,int(i%71==70),5000+i,"DEMO 软件历史特征测试，非真实 BTS / 论文数据"])
    return rows

if __name__=="__main__":
    path=Path(__file__).resolve().parents[1]/"data/demo/demo_historical_flights.csv"
    if path.exists(): raise SystemExit("历史 DEMO 已存在，拒绝覆盖")
    print(create_demo(path))
