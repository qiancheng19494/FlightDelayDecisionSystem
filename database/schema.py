"""第一版持久化结构；业务写入将在后续阶段接入。"""
SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset_records (
 id INTEGER PRIMARY KEY, file_name TEXT NOT NULL, file_path TEXT NOT NULL,
 record_count INTEGER NOT NULL DEFAULT 0 CHECK(record_count >= 0),
 date_start TEXT, date_end TEXT, field_count INTEGER,
 imported_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS prediction_records (
 id INTEGER PRIMARY KEY, predicted_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
 flight_date TEXT NOT NULL, carrier TEXT NOT NULL, origin TEXT NOT NULL, destination TEXT NOT NULL,
 model_name TEXT NOT NULL, model_version TEXT NOT NULL,
 threshold REAL NOT NULL CHECK(threshold BETWEEN 0 AND 1),
 probability REAL NOT NULL CHECK(probability BETWEEN 0 AND 1),
 predicted_class INTEGER NOT NULL CHECK(predicted_class IN (0,1)), input_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON prediction_records(flight_date);
CREATE TABLE IF NOT EXISTS model_records (
 id INTEGER PRIMARY KEY, model_name TEXT NOT NULL, model_version TEXT NOT NULL,
 file_path TEXT NOT NULL, metadata_json TEXT NOT NULL,
 registered_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS system_logs (
 id INTEGER PRIMARY KEY, event_time TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
 level TEXT NOT NULL, event TEXT NOT NULL, details TEXT
);
"""
