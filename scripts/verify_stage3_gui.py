"""第三阶段真实 Qt 后台构建、查询、冷启动、审计及报告验收与截图。"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PySide6.QtCore import Qt,QDate,QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QFileDialog
from core.bootstrap import initialize
from core.historical_features import json_safe
from app.main_window import MainWindow


def verify():
    app=QApplication.instance() or QApplication([])
    app.setStyle("Fusion");app.setStyleSheet((ROOT/"resources/styles/main.qss").read_text(encoding="utf-8"))
    errors=[];previous=sys.excepthook
    sys.excepthook=lambda *args:errors.append(str(args[1]))
    with tempfile.TemporaryDirectory(prefix="stage3_gui_") as folder:
        root=Path(folder);(root/"config").mkdir()
        shutil.copy2(ROOT/"config/default_config.json",root/"config/default_config.json")
        context=initialize(root)
        record=context.datasets.import_dataset(ROOT/"data/demo/demo_historical_flights.csv")
        window=MainWindow(context);window.resize(1440,900);window.show()
        page=window.pages[3]
        window.navigation.setCurrentRow(3)
        def process():
            app.processEvents()
            if errors:raise AssertionError(errors)
        def wait():
            deadline=time.monotonic()+90
            while page.task.busy:
                process();QTest.qWait(5)
                if time.monotonic()>deadline:raise AssertionError("history GUI task timed out")
            process()
        def shot(name):
            process();window.stack.currentWidget().verticalScrollBar().setValue(0);QTest.qWait(100)
            assert window.grab().save(str(ROOT/"output/figures"/name))
        beats=[]
        timer=QTimer();timer.setInterval(20);timer.timeout.connect(lambda:beats.append(time.perf_counter()))
        timer.start();start=time.perf_counter()
        QTest.mouseClick(page.build_button,Qt.LeftButton);wait()
        timer.stop()
        assert page.engine is not None
        assert page.engine.metadata["source_rows"]==540
        assert page.engine.metadata["validation"]["audit"]["passed"]
        shot("stage3_history_overview.png")
        page.tabs.setCurrentIndex(1);wait()
        for entity in ["airline","origin","destination","route"]:
            page.entity_selector.setCurrentIndex(page.entity_selector.findData(entity))
            QTest.mouseClick(page.entity_button,Qt.LeftButton);wait()
            assert page.entity_table.model().rowCount()>0
        shot("stage3_history_entity.png")
        page.tabs.setCurrentIndex(2)
        def query(date,airline,origin,destination):
            page.query_date.setDate(QDate.fromString(date,"yyyy-MM-dd"))
            page.query_airline.setText(airline);page.query_origin.setText(origin);page.query_destination.setText(destination)
            with patch("core.data_loader.FlightDataLoader.iter_chunks",side_effect=AssertionError("query must not scan CSV")):
                QTest.mouseClick(page.query_button,Qt.LeftButton);wait()
            return page.query_result
        normal=query("2024-01-15","AA","AAA","BBB")
        assert normal.features["hist_route_count"]>0
        shot("stage3_history_query.png")
        cold=query("2024-01-15","ZX","AAA","DDD")
        assert cold.features["hist_route_count"]==0 and cold.features["cold_start_route"]==1
        assert cold.features["hist_airline_origin_count"]==0
        assert cold.metadata["hist_route_fallback_source"]=="ORIGIN_DESTINATION"
        shot("stage3_cold_start.png")
        first=query("2024-01-01","AA","AAA","BBB")
        assert page.query_table.model().frame["延误率"].eq("暂无历史").all()
        assert page.query_table.model().frame["平均延误"].eq("暂无历史").all()
        shot("stage3_first_day_no_history.png")
        QTest.mouseClick(page.audit_button,Qt.LeftButton);wait()
        assert page.audit_table.model().frame["状态"].eq("PASS").all()
        shot("stage3_leakage_audit.png")
        output=root/"history_report.md"
        with patch.object(QFileDialog,"getSaveFileName",return_value=(str(output),"Markdown")):
            QTest.mouseClick(page.report_button,Qt.LeftButton);wait()
        assert "date < target_date" in output.read_text(encoding="utf-8")
        # 切换页面及加载缓存不能重新扫描源文件。
        with patch("core.data_loader.FlightDataLoader.iter_chunks",side_effect=AssertionError("cache reload rescanned CSV")):
            QTest.mouseClick(page.refresh_button,Qt.LeftButton);wait()
        for size in [(1440,900),(1366,768)]:
            window.resize(*size)
            for index in range(15):
                window.navigation.setCurrentRow(index);process()
                assert window.stack.currentIndex()==index
                assert window.stack.currentWidget().horizontalScrollBar().maximum()==0,(size,index)
        gaps=[b-a for a,b in zip([start]+beats,beats)]
        result={"visible_gui":window.isVisible(),"navigation_pages":15,"background_heartbeat_count":len(beats),"max_heartbeat_gap_seconds":round(max(gaps),4) if gaps else None,"history_rows":page.engine.metadata["source_rows"],"normal_query":json_safe({"features":normal.features,"metadata":normal.metadata}),"cold_start_query":json_safe({"features":cold.features,"metadata":cold.metadata}),"first_day_query":json_safe(first.features),"audit":page.engine.metadata["validation"]["audit"],"query_and_cache_reload_without_csv_scan":True,"report_export":True,"screenshots":6}
        assert len(beats)>0
        window.close();process()
        for handler in list(context.logger.handlers):handler.close();context.logger.removeHandler(handler)
    sys.excepthook=previous
    (ROOT/"output/reports/stage3_gui_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ["normal_query","cold_start_query","first_day_query"]},ensure_ascii=False))

if __name__=="__main__":verify()
