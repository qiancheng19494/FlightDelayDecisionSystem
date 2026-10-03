"""Exercise actual Stage5.2 GUI, using the bounded real sample already validated."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt,QTimer
from PySide6.QtTest import QTest
from core.bootstrap import initialize
from app.main_window import MainWindow
from core.ml.model_registry import atomic_json

def verify():
    app=QApplication.instance() or QApplication([]);app.setStyle("Fusion");app.setStyleSheet((ROOT/"resources/styles/main.qss").read_text(encoding="utf-8"))
    smoke=json.loads((ROOT/"output/reports/stage5_2_real_bts_validation.json").read_text(encoding="utf-8"));assert smoke["status"]=="PASS"
    context=initialize(ROOT);w=MainWindow(context);w.resize(1440,1050);w.show();errors=[];old=sys.excepthook;sys.excepthook=lambda *args:errors.append(str(args[1]))
    def process():
        app.processEvents()
        if errors:raise AssertionError(errors)
    def wait(task):
        deadline=time.monotonic()+90
        while task.busy:
            process();QTest.qWait(10)
            if time.monotonic()>deadline:raise AssertionError("GUI timeout")
        process()
    screenshots=[]
    def shot(name):
        process();w.stack.currentWidget().verticalScrollBar().setValue(0);QTest.qWait(120)
        assert w.grab().save(str(ROOT/"output/figures"/name));screenshots.append(name)
    beats=[];timer=QTimer();timer.setInterval(20);timer.timeout.connect(lambda:beats.append(time.perf_counter()));timer.start();start=time.perf_counter()
    data=w.pages[1];wait(data.task);w.navigation.setCurrentRow(1);data.refresh(smoke["dataset_id"]);wait(data.task)
    QTest.mouseClick(data.compatibility_button,Qt.LeftButton);wait(data.task);assert data.compatibility_result["compatible"]
    shot("stage5_2_bts_compatibility.png")
    quality=w.pages[2];w.navigation.setCurrentRow(2);quality.selector.setCurrentIndex(quality.selector.findData(smoke["dataset_id"]));wait(quality.task)
    QTest.mouseClick(quality.analyze_button,Qt.LeftButton);wait(quality.task);shot("stage5_2_real_bts_summary.png")
    w.navigation.setCurrentRow(6);p=w.pages[6];p.mode_tabs.setCurrentIndex(0)
    for box in (p.dataset,p.history,p.schedule):box.setCurrentIndex(box.findData(smoke["dataset_id"]))
    QTest.mouseClick(p.build_button,Qt.LeftButton);wait(p.task);assert p.result is not None and p.result.X.shape==(100,80)
    QTest.mouseClick(p.audit_button,Qt.LeftButton);p.tabs.setCurrentIndex(4);assert p.audit.model().frame["结果"].eq("PASS").all();shot("stage5_2_real_feature_audit.png")
    p.mode_tabs.setCurrentIndex(2);m=p.management;m.refresh();m.selector.setCurrentIndex(m.selector.findData(smoke["models"][0]["metadata"]["model_id"]))
    assert "MIXED" in m.protocol_audit_view.toPlainText();shot("stage5_2_protocol_view.png")
    QTest.mouseClick(m.protocol_button,Qt.LeftButton);assert "2024-12-14" in m.details.toPlainText();shot("stage5_2_protocol_audit.png")
    QTest.mouseClick(m.load_button,Qt.LeftButton);wait(m.task);assert m.loaded.metadata["status"]=="REAL_SMOKE"
    timer.stop()
    for size in [(1440,900),(1366,768)]:
        w.resize(*size)
        for i in range(15):w.navigation.setCurrentRow(i);process();assert w.stack.currentWidget().horizontalScrollBar().maximum()==0,(size,i)
    gaps=[b-a for a,b in zip([start]+beats,beats)]
    result=dict(passed=True,visible_gui=w.isVisible(),dataset_id=smoke["dataset_id"],bts_compatibility=True,protocol_display=True,real_feature_audit=True,model_load=True,pages=15,screenshots=screenshots,heartbeats=len(beats),max_gap_seconds=max(gaps))
    atomic_json(ROOT/"output/reports/stage5_2_gui_verification.json",result)
    w.close();process();sys.excepthook=old
    for h in list(context.logger.handlers):h.close();context.logger.removeHandler(h)
    print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":verify()
