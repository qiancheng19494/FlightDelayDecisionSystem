"""线程池任务和统一 GUI 错误转换。"""
import logging
import sqlite3
import threading
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

class TaskSignals(QObject):
    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(object)
    error = Signal(str)
    settled = Signal()
    stage_changed = Signal(str)
    cancelled = Signal()

class TaskWorker(QRunnable):
    def __init__(self, function, title: str, with_stage=False):
        super().__init__()
        self.function, self.title = function, title
        self.with_stage = with_stage
        self.signals = TaskSignals()
        self.cancel_event = threading.Event()

    @Slot()
    def run(self):
        self.signals.status.emit(self.title)
        try:
            arguments = [self.signals.progress.emit, self.cancel_event.is_set]
            if self.with_stage:
                arguments.append(self.signals.stage_changed.emit)
            result = self.function(*arguments)
            self.signals.finished.emit(result)
        except Exception as exc:
            if self.cancel_event.is_set():
                self.signals.cancelled.emit()
                return
            logging.getLogger("flight_delay").exception("后台任务失败：%s", self.title)
            if isinstance(exc, PermissionError):
                message = "没有文件访问权限或文件被占用，请关闭占用程序并检查目录权限。"
            elif isinstance(exc, sqlite3.Error):
                message = "数据库写入或读取失败，请检查磁盘空间和数据库是否被占用。"
            elif isinstance(exc, OSError):
                message = "文件访问失败，请检查路径、源文件及磁盘空间。"
            elif isinstance(exc, ValueError):
                message = str(exc)
            else:
                message = "任务未完成，请查看日志中的详细原因。"
            self.signals.error.emit(message)
        finally:
            self.signals.settled.emit()

class TaskController(QObject):
    """持有任务生命周期；所有回调通过主线程 QObject 槽执行。"""
    progress = Signal(int, int)
    status = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)
    busy_changed = Signal(bool)
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.busy = False
        self.worker = None
        self.callback = None
        self.result = None
        self.has_result = False

    def start(self, function, title, callback, with_stage=False):
        if self.busy:
            return False
        self.busy, self.callback = True, callback
        self.has_result = False
        self.worker = TaskWorker(function, title, with_stage)
        self.worker.signals.stage_changed.connect(self.status)
        self.worker.signals.cancelled.connect(self.cancelled)
        self.worker.signals.progress.connect(self.progress)
        self.worker.signals.status.connect(self.status)
        self.worker.signals.finished.connect(self._success)
        self.worker.signals.error.connect(self.failed)
        self.worker.signals.settled.connect(self._settled)
        self.busy_changed.emit(True)
        QThreadPool.globalInstance().start(self.worker)
        return True

    @Slot(object)
    def _success(self, result):
        self.result = result
        self.has_result = True

    @Slot()
    def _settled(self):
        self.busy = False
        self.worker = None
        self.busy_changed.emit(False)
        callback, result, has_result = self.callback, self.result, self.has_result
        self.callback, self.result, self.has_result = None, None, False
        if has_result:
            if callback:
                callback(result)
            self.succeeded.emit(result)

    def cancel(self):
        if self.worker:
            self.worker.cancel_event.set()
