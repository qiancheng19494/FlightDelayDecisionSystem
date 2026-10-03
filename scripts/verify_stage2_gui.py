"""实际 Qt 控件验收：仅文件对话框返回值自动化，业务和后台任务均真实运行。"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from app.main_window import MainWindow
from core.bootstrap import initialize
from scripts.create_demo_data import create_demo


def verify():
    app=QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet((ROOT/"resources/styles/main.qss").read_text(encoding="utf-8"))
    screenshots=ROOT/"output/figures"
    screenshots.mkdir(parents=True,exist_ok=True)
    errors=[]
    previous_hook=sys.excepthook
    sys.excepthook=lambda *args: errors.append(str(args[1]))
    with tempfile.TemporaryDirectory(prefix="stage2_gui_") as directory:
        root=Path(directory)
        (root/"config").mkdir()
        shutil.copy2(ROOT/"config/default_config.json",root/"config/default_config.json")
        context=initialize(root)
        window=MainWindow(context)
        window.resize(1440,900)
        window.show()
        def process():
            app.processEvents()
            if errors: raise AssertionError(errors)
        def wait_idle(page,timeout=90):
            deadline=time.monotonic()+timeout
            while page.task.busy:
                process()
                if time.monotonic()>deadline: raise AssertionError("GUI task timed out")
                QTest.qWait(5)
            process()
        def screenshot(name):
            process()
            QTest.qWait(80)
            assert window.grab().save(str(screenshots/name))
        process()
        manager,quality=window.pages[1],window.pages[2]
        window.navigation.setCurrentRow(1)
        with patch.object(QFileDialog,"getOpenFileName",return_value=(str(ROOT/"data/demo/demo_bts_flights.csv"),"CSV")), patch.object(QMessageBox,"question",return_value=QMessageBox.Yes):
            QTest.mouseClick(manager.import_button,Qt.LeftButton)
            wait_idle(manager)
        assert len(context.datasets.datasets())==1
        assert context.datasets.datasets()[0]["row_count"]==202
        assert manager.preview_table.model().rowCount()==100
        assert "成功" in manager.status.text()
        screenshot("stage2_import_success.png")
        screenshot("stage2_data_manager.png")
        QTest.mouseClick(manager.next_button,Qt.LeftButton)
        wait_idle(manager)
        assert manager.page==2
        assert "101–200" in manager.page_label.text()
        screenshot("stage2_data_preview.png")
        QTest.mouseClick(manager.next_button,Qt.LeftButton)
        wait_idle(manager)
        assert manager.preview_table.model().rowCount()==2
        assert not manager.next_button.isEnabled()
        window.navigation.setCurrentRow(0)
        screenshot("stage2_dashboard.png")
        window.navigation.setCurrentRow(2)
        QTest.mouseClick(quality.analyze_button,Qt.LeftButton)
        wait_idle(quality)
        assert quality.report["row_count"]==202
        assert quality.report["duplicate_rows"]==1
        assert len(quality.charts.figure.axes)==2
        screenshot("stage2_data_quality.png")
        quality.tabs.setCurrentIndex(1)
        screenshot("stage2_quality_charts.png")
        quality.tabs.setCurrentIndex(2)
        screenshot("stage2_quality_issues.png")
        report_path=root/"quality_export.json"
        with patch.object(QFileDialog,"getSaveFileName",return_value=(str(report_path),"JSON")):
            QTest.mouseClick(quality.export_button,Qt.LeftButton)
            wait_idle(quality)
        assert json.loads(report_path.read_text(encoding="utf-8"))["row_count"]==202
        invalid=root/"invalid.csv"
        invalid.write_text("FL_DATE\n2024-01-01\n",encoding="utf-8")
        window.navigation.setCurrentRow(1)
        with patch.object(QMessageBox,"warning") as warning:
            manager.begin_import(invalid)
            wait_idle(manager)
            assert warning.called and "airline" in manager.status.text()
        assert manager.import_button.isEnabled()
        assert len(context.datasets.datasets())==1
        # 十万行 GUI 真实后台导入，同时主线程计时器持续触发。
        large=root/"demo_100k.csv"
        create_demo(large,100000,False)
        beats=[]
        timer=QTimer();timer.setInterval(20);timer.timeout.connect(lambda:beats.append(time.perf_counter()))
        timer.start()
        start=time.perf_counter()
        with patch.object(QMessageBox,"question",return_value=QMessageBox.Yes):
            manager.begin_import(large)
            wait_idle(manager)
        elapsed=time.perf_counter()-start
        timer.stop()
        assert len(beats)>5, "GUI heartbeat stalled"
        gaps=[b-a for a,b in zip([start]+beats,beats+[time.perf_counter()])]
        assert max(gaps)<2, "GUI stalled for >=2 seconds"
        assert context.datasets.datasets()[0]["row_count"]==100000
        # 全部导航和两种分辨率保留可用。
        for size in [(1440,900),(1366,768)]:
            window.resize(*size)
            for index in range(15):
                window.navigation.setCurrentRow(index)
                process()
                assert window.stack.currentIndex()==index
                assert window.stack.currentWidget().horizontalScrollBar().maximum()==0,(size,index)
        result={"gui_import":True,"pagination":[100,100,2],"quality_export":True,"missing_field_error_recovered":True,"navigation_pages":15,"benchmark_rows":100000,"background_import_seconds":round(elapsed,3),"gui_heartbeat_count":len(beats),"max_heartbeat_gap_seconds":round(max(gaps),3),"screenshots":7}
        window.close()
        process()
        for handler in list(context.logger.handlers):
            handler.close();context.logger.removeHandler(handler)
    sys.excepthook=previous_hook
    (ROOT/"output/reports/stage2_gui_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    verify()
