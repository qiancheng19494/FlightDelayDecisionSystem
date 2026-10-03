"""生成确定性的 202 行软件测试 DEMO，不是论文或真实航班数据。"""
import csv
from datetime import date, timedelta
from pathlib import Path

HEADER = ["FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE", "DEP_DELAY", "CANCELLED", "OP_CARRIER_FL_NUM", "DEMO_NOTICE"]

def create_demo(path: Path, rows: int = 200, anomalies: bool = True) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        first = None
        for i in range(rows):
            row = [(date(2024, 1, 1) + timedelta(days=i%90)).isoformat(), ["AA", "DL", "UA"][i%3], ["JFK", "LAX", "ORD"][i%3], ["ATL", "SFO", "DFW"][i%3], [5, 730, 1530, 2400][i%4], 1800, 140, 900+i%20, [-5, 0, 14.9, 15, 15.1, 40][i%6], int(i%17==0), 1000+i, "DEMO 软件测试数据，非论文真实实验数据"]
            if anomalies:
                if i == 5: row[0] = "invalid-date"
                if i == 7: row[2] = ""
                if i == 8: row[3] = ""
                if i == 9: row[1] = ""
                if i == 10: row[3] = row[2]
                if i == 11: row[7] = -20
                if i == 12: row[6] = 0
                if i == 13: row[9] = 3
                if i == 14: row[8] = "bad-number"
                if i == 15: row[8] = ""
                if i == 16: row[4] = 2365
            if first is None:
                first = row.copy()
            writer.writerow(row)
        if anomalies and rows:
            writer.writerow(first)
            writer.writerow([""] * len(HEADER))
    return rows + (2 if rows and anomalies else 0)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    target = root / "data/demo/demo_bts_flights.csv"
    if target.exists():
        raise SystemExit("DEMO 已存在，拒绝覆盖。")
    print(f"DEMO rows: {create_demo(target)}")
