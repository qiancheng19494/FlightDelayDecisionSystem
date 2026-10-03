"""真实后台预览/导出/错误恢复，截图不硬编码特征值。"""
import json,shutil,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication,QFileDialog,QMessageBox
from PySide6.QtCore import QTimer,Qt
from PySide6.QtTest import QTest
from core.bootstrap import initialize
from core.historical_features import HistoricalFeatureEngine
from app.main_window import MainWindow

def verify():
    app=QApplication.instance() or QApplication([]);app.setStyle("Fusion");app.setStyleSheet((ROOT/"resources/styles/main.qss").read_text(encoding="utf-8"))
    errors=[];old=sys.excepthook;sys.excepthook=lambda *args:errors.append(str(args[1]))
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/"config").mkdir();shutil.copy2(ROOT/"config/default_config.json",root/"config/default_config.json")
        context=initialize(root);record=context.datasets.import_dataset(ROOT/"data/demo/demo_schedule_flights.csv")
        h=HistoricalFeatureEngine(context);h.build_history_cache(record["id"])
        w=MainWindow(context);w.resize(1440,900);w.show();w.navigation.setCurrentRow(6);p=w.pages[6]
        def process():
            app.processEvents()
            if errors:raise AssertionError(errors)
        def wait():
            end=time.monotonic()+90
            while p.task.busy:
                process();QTest.qWait(5)
                if time.monotonic()>end:raise AssertionError("GUI timeout")
            process()
        p.history.setCurrentIndex(p.history.findData(record["id"]));p.schedule.setCurrentIndex(p.schedule.findData(record["id"]))
        beats=[];timer=QTimer();timer.setInterval(10);timer.timeout.connect(lambda:beats.append(time.perf_counter()));timer.start()
        QTest.mouseClick(p.build_button,Qt.LeftButton);wait();timer.stop()
        assert p.result is not None and p.result.X.shape==(8,80)
        assert p.result.X.origin_hour_scheduled_departures.iloc[0]==3
        for index,name in [(0,"overview"),(1,"groups"),(2,"preview"),(3,"combination_registry"),(4,"audit"),(5,"congestion_example")]:
            if index==4:QTest.mouseClick(p.audit_button,Qt.LeftButton)
            p.tabs.setCurrentIndex(index);process();w.stack.currentWidget().verticalScrollBar().setValue(0);QTest.qWait(100)
            filename="stage4_"+(name if name in ("combination_registry","congestion_example") else "feature_"+name)+".png"
            assert w.grab().save(str(ROOT/"output/figures"/filename))
        assert p.audit.model().frame["结果"].eq("PASS").all()
        with patch.object(QFileDialog,"getExistingDirectory",return_value=str(root/"export")):
            QTest.mouseClick(p.export_button,Qt.LeftButton);wait()
        folders=list((root/"export").glob("features_*"));assert len(folders)==1
        assert (folders[0]/"X_features.csv").exists() and (folders[0]/"y_labels.csv").exists()
        p.schedule.setCurrentIndex(0);QTest.mouseClick(p.build_button,Qt.LeftButton);wait();assert p.result.X.congestion_context_available.eq(0).all()
        p.history.setCurrentIndex(0)
        with patch.object(QMessageBox,"warning") as warning:
            QTest.mouseClick(p.build_button,Qt.LeftButton);wait();assert warning.called and not p.task.busy
        for size in [(1440,900),(1366,768)]:
            w.resize(*size)
            for index in range(15):
                w.navigation.setCurrentRow(index);process();assert w.stack.currentIndex()==index;assert w.stack.currentWidget().horizontalScrollBar().maximum()==0,(size,index)
        result={"visible_gui":w.isVisible(),"pages":15,"rows":8,"columns":80,"screenshots":6,"background_heartbeats":len(beats),"sample_export":True,"missing_context_nan":True,"error_recovery":True,"feature_audit_passed":True}
        w.close();process()
        for handler in list(context.logger.handlers):handler.close();context.logger.removeHandler(handler)
    sys.excepthook=old
    (ROOT/"output/reports/stage4_gui_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(result))
if __name__=="__main__":verify()
