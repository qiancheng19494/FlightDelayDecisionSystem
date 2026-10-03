"""航班运行数据智能分析与延误预测系统 V1.0 启动入口。"""
import sys
import traceback
from pathlib import Path

def main():
    """无模型、无数据时也正常启动；启动错误转为可理解的提示。"""
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        print("缺少桌面依赖，请运行：python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    from app.constants import WINDOW_TITLE
    application = QApplication(sys.argv)
    application.setApplicationName(WINDOW_TITLE)
    root = Path(__file__).resolve().parent
    context = None
    try:
        from core.bootstrap import initialize
        from app.main_window import MainWindow
        context = initialize(root)
        application.setStyle("Fusion")
        application.setStyleSheet((root / "resources/styles/main.qss").read_text(encoding="utf-8"))
        def report_exception(kind, value, tb):
            context.logger.error("界面异常", exc_info=(kind, value, tb))
            QMessageBox.critical(None, "操作未完成", f"操作发生错误：{value}\n详细信息已记录到日志。")
        sys.excepthook = report_exception
        window = MainWindow(context)
        window.show()
        context.logger.info("主窗口显示成功")
        return application.exec()
    except Exception as exc:
        if context:
            context.logger.exception("启动失败")
        else:
            traceback.print_exc()
        QMessageBox.critical(None, "启动失败", f"系统无法启动：{exc}\n请检查项目文件、配置及目录写入权限。")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
