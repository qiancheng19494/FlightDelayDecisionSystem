"""SQLite 连接按操作关闭，可安全供后续后台任务调用。"""
import sqlite3
from contextlib import contextmanager
from database.schema import SCHEMA
from database.dataset_repository import migrate_datasets

class DatabaseManager:
    def __init__(self, path):
        self.path = path

    @contextmanager
    def connection(self):
        """成功提交，失败回滚，确保连接关闭。"""
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self):
        """幂等初始化，不删除或重建已有业务表。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > 1:
                raise RuntimeError("数据库版本高于本软件支持版本，请使用匹配的软件")
            conn.executescript(SCHEMA)
            conn.execute("PRAGMA user_version = 1")
            conn.execute("BEGIN")
            migrate_datasets(conn)

    def log_event(self, level, event, details=""):
        with self.connection() as conn:
            conn.execute("INSERT INTO system_logs(level,event,details) VALUES (?,?,?)", (level,event,details))

    def summary(self):
        """仅聚合已登记数据；无数据不计算延误率等业务指标。"""
        with self.connection() as conn:
            row = conn.execute("SELECT COUNT(*) datasets, COALESCE(SUM(record_count),0) flights, MIN(date_start) start, MAX(date_end) end FROM dataset_records").fetchone()
            result = dict(row)
            result["predictions"] = conn.execute("SELECT COUNT(*) FROM prediction_records").fetchone()[0]
            return result

    def recent(self, table, limit=5):
        if table not in ("dataset_records", "prediction_records"):
            raise ValueError("不支持的数据表")
        with self.connection() as conn:
            return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))]
