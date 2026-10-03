"""第一阶段实际基础设施回归测试。"""
import json
import logging
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from core.bootstrap import initialize
from core.config_manager import ConfigManager
from database.db_manager import DatabaseManager

SOURCE = Path(__file__).resolve().parents[1]

class InfrastructureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="航班测试_")
        self.root = Path(self.temp.name)
        (self.root / "config").mkdir()
        shutil.copy2(SOURCE / "config/default_config.json", self.root / "config/default_config.json")

    def tearDown(self):
        logger = logging.getLogger("flight_delay")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        self.temp.cleanup()

    def test_first_run_without_assets(self):
        context = initialize(self.root)
        self.assertEqual(context.database.summary()["flights"], 0)
        self.assertEqual(context.database.summary()["predictions"], 0)
        for folder in ["data/raw", "logs", "output/reports", "models"]:
            self.assertTrue((self.root / folder).is_dir())
        self.assertIn("初始化成功", (self.root / "logs/application.log").read_text(encoding="utf-8"))
        self.assertEqual(list((self.root / "models").iterdir()), [])

    def test_config_roundtrip(self):
        config = ConfigManager(self.root)
        config.save({"default_threshold": 0.37, "data_directory": "data/中文"})
        self.assertEqual(ConfigManager(self.root).get("default_threshold"), 0.37)
        self.assertEqual(ConfigManager(self.root).get("data_directory"), "data/中文")

    def test_bad_config_preserved(self):
        path = self.root / "config/user_config.json"
        path.write_text("broken-json", encoding="utf-8")
        config = ConfigManager(self.root)
        self.assertEqual(config.get("default_threshold"), 0.5)
        self.assertTrue(config.warnings)
        self.assertEqual(path.read_text(encoding="utf-8"), "broken-json")

    def test_config_rejects_invalid_values(self):
        config = ConfigManager(self.root)
        for update in [{"default_threshold": 1.1}, {"default_threshold": float("nan")}, {"page_size": 0}, {"data_directory": "../escape"}, {"log_level": "BOGUS"}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                config.save(update)
        self.assertFalse(config.path.exists())

    def test_database_idempotent_and_constraints(self):
        database = DatabaseManager(self.root / "database/test.sqlite3")
        database.initialize()
        database.log_event("INFO", "中文事件", "测试")
        database.initialize()
        with database.connection() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM system_logs").fetchone()[0], 1)
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({"dataset_records", "prediction_records", "model_records", "system_logs"} <= tables)
        with self.assertRaises(ValueError):
            database.recent("invalid_table")

    def test_database_transaction_rolls_back(self):
        database = DatabaseManager(self.root / "rollback.sqlite3")
        database.initialize()
        with self.assertRaises(sqlite3.IntegrityError):
            with database.connection() as conn:
                conn.execute("INSERT INTO dataset_records(file_name,file_path,record_count) VALUES (?,?,?)", ("中文.csv", "data/raw/中文.csv", 10))
                conn.execute("INSERT INTO dataset_records(file_name,file_path,record_count) VALUES (?,?,?)", ("bad.csv", "data/raw/bad.csv", -1))
        self.assertEqual(database.summary()["datasets"], 0)

    def test_future_database_version_rejected(self):
        database = DatabaseManager(self.root / "future.sqlite3")
        with database.connection() as conn:
            conn.execute("PRAGMA user_version=2")
        with self.assertRaises(RuntimeError):
            database.initialize()

if __name__ == "__main__":
    unittest.main()
