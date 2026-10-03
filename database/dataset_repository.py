"""数据集元信息存储，不保存逐条航班数据。"""
import json
COLUMNS = {
 "dataset_name": "TEXT", "dataset_type": "TEXT NOT NULL DEFAULT 'UNKNOWN'",
 "file_size": "INTEGER", "mtime_ns": "INTEGER", "row_count": "INTEGER", "column_count": "INTEGER",
 "start_date": "TEXT", "end_date": "TEXT", "airline_count": "INTEGER", "origin_count": "INTEGER",
 "destination_count": "INTEGER", "cancelled_count": "INTEGER", "valid_labeled_count": "INTEGER",
 "delay_count": "INTEGER", "delay_rate": "REAL", "encoding": "TEXT",
 "status": "TEXT NOT NULL DEFAULT 'LEGACY_NEEDS_SCAN'", "mapping_json": "TEXT",
 "cache_path": "TEXT", "is_demo": "INTEGER NOT NULL DEFAULT 0"
}

def migrate_datasets(connection) -> None:
    """保留 V1 表和 PRAGMA 版本约定，用独立模块版本表管理附加迁移。"""
    connection.execute("CREATE TABLE IF NOT EXISTS schema_versions (module TEXT PRIMARY KEY, version INTEGER NOT NULL)")
    row = connection.execute("SELECT version FROM schema_versions WHERE module='datasets'").fetchone()
    if row and row[0] > 2:
        raise RuntimeError("数据集结构版本高于当前软件支持版本")
    existing = {row[1] for row in connection.execute("PRAGMA table_info(dataset_records)")}
    for name, definition in COLUMNS.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE dataset_records ADD COLUMN {name} {definition}")
    connection.execute("UPDATE dataset_records SET dataset_name=COALESCE(dataset_name,file_name), row_count=COALESCE(row_count,record_count), column_count=COALESCE(column_count,field_count), start_date=COALESCE(start_date,date_start), end_date=COALESCE(end_date,date_end)")
    connection.execute("INSERT OR REPLACE INTO schema_versions(module,version) VALUES ('datasets',2)")

class DatasetRepository:
    def __init__(self, database):
        self.database = database

    def save(self, quality: dict, cache_path: str, is_demo: bool, dataset_id: int | None = None) -> int:
        """一次事务写完整摘要，并保持第一阶段字段可读取。"""
        keys = ["file_name", "file_path", "file_size", "mtime_ns", "row_count", "column_count", "start_date", "end_date", "dataset_type", "airline_count", "origin_count", "destination_count", "cancelled_count", "valid_labeled_count", "delay_count", "delay_rate", "encoding"]
        record = {key: quality[key] for key in keys}
        record.update(dataset_name=("DEMO / " if is_demo else "") + quality["file_name"], record_count=quality["row_count"], field_count=quality["column_count"], date_start=quality["start_date"], date_end=quality["end_date"], status="READY", mapping_json=json.dumps(quality["mapping"], ensure_ascii=False), cache_path=cache_path, is_demo=int(is_demo))
        with self.database.connection() as conn:
            if dataset_id is None:
                names = list(record)
                cursor = conn.execute(f"INSERT INTO dataset_records ({','.join(names)}) VALUES ({','.join('?' for _ in names)})", list(record.values()))
                return cursor.lastrowid
            cursor = conn.execute("UPDATE dataset_records SET " + ",".join(f"{key}=?" for key in record) + " WHERE id=?", [*record.values(), dataset_id])
            if cursor.rowcount != 1:
                raise ValueError("数据集记录不存在")
            return dataset_id

    def list(self) -> list[dict]:
        with self.database.connection() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM dataset_records ORDER BY id DESC")]

    def get(self, dataset_id: int) -> dict:
        with self.database.connection() as conn:
            row = conn.execute("SELECT * FROM dataset_records WHERE id=?", (dataset_id,)).fetchone()
        if row is None:
            raise ValueError("数据集记录不存在")
        return dict(row)

    def set_status(self, dataset_id: int, status: str) -> None:
        if status not in {"READY", "SOURCE_MISSING", "SOURCE_CHANGED", "ARCHIVED", "LEGACY_NEEDS_SCAN"}:
            raise ValueError("不支持的数据集状态")
        with self.database.connection() as conn:
            conn.execute("UPDATE dataset_records SET status=? WHERE id=?", (status, dataset_id))
