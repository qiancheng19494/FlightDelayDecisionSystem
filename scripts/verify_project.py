"""在隔离目录验证 15 页导航和不同窗口尺寸；不生成业务数据。"""
import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtWidgets import QApplication
from app.main_window import MainWindow
from app.constants import WINDOW_TITLE
from core.bootstrap import initialize

def verify(screenshot_directory=None):
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet((ROOT / "resources/styles/main.qss").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="flight_gui_") as directory:
        root = Path(directory)
        (root / "config").mkdir()
        shutil.copy2(ROOT / "config/default_config.json", root / "config/default_config.json")
        context = initialize(root)
        window = MainWindow(context)
        window.show()
        app.processEvents()
        assert window.isVisible()
        assert window.windowTitle() == WINDOW_TITLE
        assert window.stack.count() == window.navigation.count() == 15
        for size in [(1440, 900), (1366, 768)]:
            window.resize(*size)
            for index in range(15):
                window.navigation.setCurrentRow(index)
                app.processEvents()
                assert window.stack.currentIndex() == index
                assert window.stack.currentWidget().isVisible()
                scroll = window.stack.currentWidget()
                assert scroll.horizontalScrollBar().maximum() == 0, (size, index)
            if screenshot_directory:
                destination = Path(screenshot_directory)
                destination.mkdir(parents=True, exist_ok=True)
                for index, name in [(0, "dashboard"), (14, "about")]:
                    window.navigation.setCurrentRow(index)
                    app.processEvents()
                    assert window.grab().save(str(destination / f"{name}_{size[0]}.png"))
        window.navigation.setCurrentRow(0)
        from PySide6.QtWidgets import QPushButton
        buttons = window.pages[0].findChildren(QPushButton)
        for button, target in zip(buttons, [1, 6, 14]):
            button.click()
            app.processEvents()
            assert window.stack.currentIndex() == target
        window.close()
        app.processEvents()
        for handler in list(context.logger.handlers):
            handler.close()
            context.logger.removeHandler(handler)
    print("PASS: visible GUI, 15 pages, dashboard shortcuts, 1440x900 / 1366x768, no model or data required")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshots")
    args = parser.parse_args()
    verify(args.screenshots)
