"""续跑接口验收：实际查询接口、NaN/回退展示、失败状态和报告来源。"""
import copy
import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from tests.data_helpers import DataTestCase
from core.historical_features import HistoricalFeatureEngine
from core.history_report import export_history_report
from app.pages.history_analysis_page import HistoryAnalysisPage

class HistoryGuiReportTests(DataTestCase):
    @classmethod
    def setUpClass(cls):
        cls.application=QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        source=self.root/"tiny.csv"
        shutil.copy2(Path(__file__).parent/"fixtures/historical_tiny.csv",source)
        record=self.context.datasets.import_dataset(source)
        self.engine=HistoricalFeatureEngine(self.context)
        self.engine.build_history_cache(record["id"])
        self.page=HistoryAnalysisPage(self.context)
        self.page.display_engine(self.engine)

    def tearDown(self):
        self.page.close();self.page.deleteLater();self.application.processEvents()
        super().tearDown()

    def query(self,date="2024-01-03",airline="AA",origin="AAA",destination="BBB"):
        return self.engine.transform_single(dict(flight_date=date,airline=airline,origin=origin,destination=destination))

    def test_first_day_gui_na_not_zero(self):
        self.page.query_ready(self.query("2024-01-01"))
        frame=self.page.query_table.model().frame
        self.assertTrue(frame["延误率"].eq("暂无历史").all())
        self.assertTrue(frame["平均延误"].eq("暂无历史").all())
        self.assertTrue(frame["自身数量"].eq(0).all())
    def test_gui_fallback_preserves_zero_count(self):
        self.page.query_ready(self.query(destination="DDD"))
        row=self.page.query_table.model().frame.iloc[4]
        self.assertEqual(row["自身数量"],0)
        self.assertEqual(row["冷启动"],1)
        self.assertEqual(row["回退来源"],"ORIGIN_DESTINATION")
        self.assertIn("最大历史来源 2024-01-02",self.page.query_notice.text())
    def test_query_button_calls_same_engine_without_csv_scan(self):
        with patch.object(self.engine,"transform_single",wraps=self.engine.transform_single) as method,patch("core.data_loader.FlightDataLoader.iter_chunks",side_effect=AssertionError("query rescanned CSV")):
            self.page.query_button.click()
            deadline=time.monotonic()+5
            while self.page.task.busy:
                self.application.processEvents();QTest.qWait(5)
                self.assertLess(time.monotonic(),deadline)
            self.assertEqual(method.call_count,1)
            self.assertEqual(self.page.query_result.features["hist_route_count"],3)
    def test_audit_gui_uses_real_result(self):
        self.assertIn("通过",self.page.audit_label.text())
        self.assertTrue(self.page.audit_table.model().frame["状态"].eq("PASS").all())
        failed=copy.deepcopy(self.engine.metadata["validation"])
        failed["audit"]["passed"]=False;failed["audit"]["check_results"][0]["passed"]=False
        self.page.display_audit(failed)
        self.assertIn("未通过",self.page.audit_label.text())
        self.assertEqual(self.page.audit_table.model().frame.iloc[0]["状态"],"FAIL")
    def test_report_markdown_and_json(self):
        md=self.root/"report.md";js=self.root/"report.json"
        export_history_report(self.engine,md);export_history_report(self.engine,js)
        text=md.read_text(encoding="utf-8")
        self.assertIn("date < target_date",text)
        self.assertIn("未找到当前版本全量测试记录",text)
        self.assertTrue(json.loads(js.read_text(encoding="utf-8"))["validation"]["audit"]["passed"])
    def test_report_does_not_forge_pass(self):
        self.engine.metadata["validation"]["audit"]["passed"]=False
        output=self.root/"failed.md"
        export_history_report(self.engine,output)
        self.assertIn("结果：FAIL",output.read_text(encoding="utf-8"))
    def test_gui_rejects_prediction_input(self):
        record=self.context.datasets.import_dataset(self.csv(rows=[["2024-01-01","AA","AAA","BBB"]],columns=["FL_DATE","AIRLINE","ORIGIN","DEST"]))
        self.page.refresh()
        self.page.selector.setCurrentIndex(self.page.selector.findData(record["id"]))
        self.assertFalse(self.page.build_button.isEnabled())
        self.assertIn("不包含真实出发延误标签",self.page.status.text())
