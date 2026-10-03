"""实际 GUI 训练四个指定 DEMO 组合，复用已存在资产，不重复训练。"""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer,Qt
from PySide6.QtTest import QTest
from core.bootstrap import initialize
from core.ml.model_manager import ModelManager
from app.main_window import MainWindow

def verify():
    app=QApplication.instance() or QApplication([]);app.setStyle("Fusion");app.setStyleSheet((ROOT/"resources/styles/main.qss").read_text(encoding="utf-8"))
    context=initialize(ROOT);w=MainWindow(context);w.resize(1440,1000);w.show();w.navigation.setCurrentRow(6);p=w.pages[6];p.mode_tabs.setCurrentIndex(1);t=p.training;m=p.management
    errors=[];old=sys.excepthook;sys.excepthook=lambda *args:errors.append(str(args[1]))
    def process():
        app.processEvents()
        if errors:raise AssertionError(errors)
    def wait(task):
        end=time.monotonic()+180
        while task.busy:
            process();QTest.qWait(10)
            if time.monotonic()>end:raise AssertionError("training GUI timeout")
        process()
    def shot(name):
        process();w.stack.currentWidget().verticalScrollBar().setValue(0);QTest.qWait(150);assert w.grab().save(str(ROOT/"output/figures"/name))
    for box in [t.dataset,t.history,t.schedule]:box.setCurrentIndex(box.findData(2))
    QTest.mouseClick(t.estimate_button,Qt.LeftButton);wait(t.task);shot("stage5_model_training.png")
    beats=[];timer=QTimer();timer.setInterval(20);timer.timeout.connect(lambda:beats.append(time.perf_counter()));timer.start();started=time.perf_counter()
    outputs=[];new_training_count=0
    for kind,combo in [("rf","COMBO_A"),("rf","COMBO_ATHC"),("xgboost","COMBO_A"),("xgboost","COMBO_ATHC")]:
        t.model.setCurrentIndex(t.model.findData(kind));t.combo.setCurrentIndex(t.combo.findData(combo))
        records=ModelManager(ROOT).registry.list_models();existing=next((r for r in records if r["model_type"]==kind and r["combination_id"]==combo and r["dataset_id"]==2 and r["status"]=="DEMO"),None)
        if existing:
            b=ModelManager(ROOT).load_model(existing["model_id"]);t.show_metadata(b.metadata,str(ROOT/"models/bundles"/b.model_id))
            result={"metadata":b.metadata,"model_path":str(ROOT/"models/bundles"/b.model_id),"serialization_equal":True}
        else:
            new_training_count+=1
            QTest.mouseClick(t.train_button,Qt.LeftButton);wait(t.task);assert t.result is not None;t_result=t.result;assert t_result["metadata"]["model_type"]==kind;result=t_result
        outputs.append(result)
        if combo=="COMBO_ATHC":shot("stage5_rf_result.png" if kind=="rf" else "stage5_xgb_result.png")
    timer.stop();shot("stage5_demo_warning.png");shot("stage5_model_validation.png")
    p.mode_tabs.setCurrentIndex(2);m.refresh();shot("stage5_model_registry.png")
    QTest.mouseClick(m.load_button,Qt.LeftButton);wait(m.task);assert m.loaded is not None
    QTest.mouseClick(m.meta_button,Qt.LeftButton);shot("stage5_model_metadata.png")
    # 不默认为用户选择“最佳”模型；设置当前模型只在隔离自动测试中验证。
    for size in [(1440,900),(1366,768)]:
        w.resize(*size)
        for index in range(15):
            w.navigation.setCurrentRow(index);process();assert w.stack.currentWidget().horizontalScrollBar().maximum()==0,(size,index)
    gaps=[b-a for a,b in zip([started]+beats,beats)]
    report={"visible_gui":w.isVisible(),"navigation_pages":15,"background_heartbeats":len(beats),"max_heartbeat_gap_seconds":max(gaps) if gaps else None,"model_load_gui":m.loaded is not None,"models":[r["metadata"]["model_id"] for r in outputs],"screenshots":7,"demo_notice_visible":True}
    report["new_training_count"]=new_training_count
    previous_path=ROOT/"output/reports/stage5_gui_verification.json"
    if previous_path.exists() and new_training_count==0:
        previous=json.loads(previous_path.read_text(encoding="utf-8"))
        report["original_training_evidence"]=previous.get("original_training_evidence",previous)
    (ROOT/"output/reports/stage5_gui_verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    w.close();process();sys.excepthook=old
    for handler in list(context.logger.handlers):handler.close();context.logger.removeHandler(handler)
    print(json.dumps(report))
if __name__=="__main__":verify()
