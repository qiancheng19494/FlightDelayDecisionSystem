"""测试数据与隔离配置。"""
import csv
import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from core.bootstrap import initialize

HEADER = ["FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "DEP_DELAY", "CANCELLED"]
ROWS = [["2024-01-01", "AA", "JFK", "LAX", "14.9", "0"], ["01/02/2024", "DL", "LAX", "JFK", "15", "0"], ["2024/02/01", "AA", "JFK", "LAX", "15.1", "1"]]

class DataTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="数据测试_")
        self.root = Path(self.temporary.name)
        (self.root / "config").mkdir()
        source = Path(__file__).resolve().parents[1]
        shutil.copy2(source / "config/default_config.json", self.root / "config/default_config.json")
        self.context = initialize(self.root)
        self.context.config.save({"csv_chunksize": 100})

    def csv(self, rows=None, columns=None, encoding="utf-8", name="input.csv"):
        path = self.root / name
        with path.open("w", encoding=encoding, newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns if columns is not None else HEADER)
            writer.writerows(rows if rows is not None else ROWS)
        return path

    def tearDown(self):
        for handler in list(self.context.logger.handlers):
            handler.close()
            self.context.logger.removeHandler(handler)
        self.temporary.cleanup()
