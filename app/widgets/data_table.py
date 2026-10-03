"""只显示受限预览页或摘要表的只读模型。"""
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import QTableView, QAbstractItemView

class FrameModel(QAbstractTableModel):
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.frame = frame

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index, role=Qt.DisplayRole):
        if role in (Qt.DisplayRole, Qt.ToolTipRole) and index.isValid():
            value = self.frame.iat[index.row(), index.column()]
            return "" if pd.isna(value) else str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(self.frame.columns[section]) if orientation == Qt.Horizontal else str(section+1)

class DataTable(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setMinimumWidth(0)
        self.set_frame(pd.DataFrame())

    def set_frame(self, frame):
        old = self.model()
        self.setModel(FrameModel(frame, self))
        if old:
            old.deleteLater()
        self.horizontalHeader().setDefaultSectionSize(140)
