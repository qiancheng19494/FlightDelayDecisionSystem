"""安全迁移、元信息 CRUD、缓存和源失效。"""
import json
import sqlite3
from unittest.mock import patch
from tests.data_helpers import DataTestCase
from database.db_manager import DatabaseManager
from database.schema import SCHEMA

class DatasetTests(DataTestCase):
    def test_write_read_update_status(self):
        service=self.context.datasets
        record=service.import_dataset(self.csv())
        self.assertEqual(service.repository.get(record["id"])["row_count"],3)
        service.repository.set_status(record["id"],"ARCHIVED")
        self.assertEqual(service.repository.get(record["id"])["status"],"ARCHIVED")
    def test_legacy_migration_preserves_rows(self):
        path=self.root/"legacy.sqlite3"
        with sqlite3.connect(path) as conn:
            conn.executescript(SCHEMA)
            conn.execute("INSERT INTO dataset_records(file_name,file_path,record_count) VALUES ('old.csv','old.csv',123)")
            conn.execute("PRAGMA user_version=1")
        conn.close()
        db=DatabaseManager(path);db.initialize();db.initialize()
        record=db.recent("dataset_records")[0]
        self.assertEqual(record["row_count"],123)
        self.assertEqual(record["status"],"LEGACY_NEEDS_SCAN")
    def test_cache_reused(self):
        service=self.context.datasets
        record=service.import_dataset(self.csv())
        with patch.object(service,"loader",side_effect=AssertionError("must not rescan")):
            self.assertEqual(service.analyze(record["id"])["row_count"],3)
    def test_source_changed_then_rescan(self):
        service=self.context.datasets;path=self.csv();record=service.import_dataset(path)
        with path.open("a",encoding="utf-8") as stream: stream.write("2024-03-01,UA,ORD,LAX,20,0\n")
        self.assertEqual(service.datasets()[0]["status"],"SOURCE_CHANGED")
        self.assertIsNone(service.cached(record))
        self.assertEqual(service.analyze(record["id"])["row_count"],4)
        self.assertEqual(service.repository.get(record["id"])["row_count"],4)
    def test_missing_source(self):
        service=self.context.datasets;path=self.csv();record=service.import_dataset(path);path.unlink()
        self.assertEqual(service.datasets()[0]["status"],"SOURCE_MISSING")
        with self.assertRaises(ValueError): service.preview(record["id"],1)
    def test_copy_mode(self):
        self.context.config.save({"copy_imported_files":True})
        path=self.csv();record=self.context.datasets.import_dataset(path)
        path.unlink()
        self.assertEqual(self.context.datasets.datasets()[0]["status"],"READY")
    def test_database_failure_no_success_or_cache(self):
        service=self.context.datasets
        with patch.object(service.repository,"save",side_effect=sqlite3.OperationalError("failure")):
            with self.assertRaises(sqlite3.OperationalError): service.import_dataset(self.csv())
        self.assertEqual(service.datasets(),[])
        self.assertEqual(list((self.root/"data/cache").iterdir()),[])
    def test_bad_cache_rescans(self):
        service=self.context.datasets;record=service.import_dataset(self.csv())
        (self.root/record["cache_path"]).write_text("bad",encoding="utf-8")
        self.assertEqual(service.analyze(record["id"])["row_count"],3)
    def test_cache_version_mismatch(self):
        service=self.context.datasets;record=service.import_dataset(self.csv())
        path=self.root/record["cache_path"]
        data=json.loads(path.read_text(encoding="utf-8"));data["quality_cache_version"]=999
        path.write_text(json.dumps(data),encoding="utf-8")
        self.assertIsNone(service.cached(record))
    def test_export_report(self):
        service=self.context.datasets;record=service.import_dataset(self.csv())
        target=self.root/"quality.json"
        service.export_quality(service.analyze(record["id"]),str(target))
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["row_count"],3)
    def test_demo_flag(self):
        record=self.context.datasets.import_dataset(self.csv(name="demo_test.csv"))
        self.assertTrue(record["is_demo"])
        self.assertIn("DEMO",record["dataset_name"])
