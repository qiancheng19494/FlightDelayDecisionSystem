"""保留原始 CSV 的分块读取及逻辑记录分页，不整表载入内存。"""
import codecs
import csv
from pathlib import Path
from collections.abc import Callable, Iterator
import pandas as pd
from core.field_mapper import FieldMapper

class DataImportError(ValueError):
    """面向用户的导入错误。"""

class FlightDataLoader:
    def __init__(self, path: Path | str, chunksize: int = 100000, encoding: str | None = None):
        self.path = Path(path).resolve()
        if chunksize <= 0:
            raise ValueError("chunksize 必须大于 0")
        self.chunksize = chunksize
        if not self.path.is_file():
            raise DataImportError("CSV 文件不存在，可能已被移动或删除。")
        if self.path.suffix.lower() != ".csv":
            raise DataImportError("请选择 .csv 文件。")
        self.encoding = encoding or self.detect_encoding()
        try:
            with self.path.open(encoding=self.encoding, newline="") as stream:
                self.columns = next(csv.reader(stream, strict=True), [])
        except (UnicodeError, csv.Error) as exc:
            raise DataImportError("无法读取 CSV 表头，请检查编码和分隔格式。") from exc
        if not self.columns or not any(c.strip() for c in self.columns):
            raise DataImportError("CSV 为空或没有有效表头。")
        self.original_columns=list(self.columns)
        self.bts_trailing_empty = (self.columns[-1] == "" and all(c.strip() for c in self.columns[:-1]) and {"FlightDate","Reporting_Airline","DepDel15","Cancelled"}.issubset(self.columns))
        if self.bts_trailing_empty:
            self.columns[-1]="__BTS_TRAILING_EMPTY__"
        if any(not c.strip() for c in self.columns) or len(set(self.columns)) != len(self.columns):
            raise DataImportError("表头含空列名或重复列名，请先修正。")
        self.mapping = FieldMapper().map_fields(self.columns)
        self.checkpoints: list[list[int]] = []
        self.rows_read = 0

    def detect_encoding(self) -> str:
        """仅探测前 256 KiB；全文件解码错误仍显式报告，不替换丢失字符。"""
        with self.path.open("rb") as stream:
            sample = stream.read(262144)
        if not sample:
            raise DataImportError("CSV 文件为空。")
        if b"\x00" in sample:
            raise DataImportError("文件包含二进制或 UTF-16 内容，请另存为 UTF-8 CSV。")
        for encoding in ["utf-8-sig", "gb18030", "cp1252"]:
            try:
                codecs.getincrementaldecoder(encoding)().decode(sample, final=False)
                return encoding
            except UnicodeError:
                continue
        raise DataImportError("无法识别编码，请将文件另存为 UTF-8 CSV。")

    def file_info(self) -> dict:
        stat = self.path.stat()
        return {"file_name": self.path.name, "file_path": str(self.path), "file_size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "encoding": self.encoding, "columns": self.columns, "mapping": self.mapping.to_dict()}

    def require_fields(self) -> None:
        if not self.mapping.is_valid:
            missing = ", ".join(self.mapping.missing_required_fields)
            conflicts = ", ".join(self.mapping.conflicts)
            raise DataImportError(f"字段检查未通过。缺失字段：{missing or '无'}；别名冲突：{conflicts or '无'}。")

    def _row(self, row: list[str], line: int) -> list[str]:
        if not row:
            return [""] * len(self.columns)
        if len(row) != len(self.columns):
            raise DataImportError(f"CSV 第 {line} 行列数与表头不一致，请检查引号和分隔符。")
        if self.bts_trailing_empty and row[-1].strip():
            raise DataImportError(f"BTS 第 {line} 行尾部无名列非空，拒绝丢弃")
        return row

    def iter_chunks(self, progress: Callable | None = None, cancelled: Callable | None = None) -> Iterator[pd.DataFrame]:
        """逐块产生原始字符串表；空行保留，畸形记录拒绝而非静默跳过。"""
        self.checkpoints, self.rows_read = [], 0
        size = max(self.path.stat().st_size, 1)
        try:
            with self.path.open(encoding=self.encoding, newline="") as stream:
                reader = csv.reader(iter(stream.readline, ""), strict=True)
                next(reader)
                done = False
                while not done:
                    self.checkpoints.append([self.rows_read, stream.tell()])
                    rows = []
                    for i in range(self.chunksize):
                        if i % 1024 == 0 and cancelled and cancelled():
                            raise DataImportError("任务已取消，未写入数据集记录。")
                        try:
                            rows.append(self._row(next(reader), reader.line_num))
                        except StopIteration:
                            done = True
                            break
                    if rows:
                        self.rows_read += len(rows)
                        if progress:
                            progress(min(90, int(stream.tell() / size * 90)), self.rows_read)
                        yield pd.DataFrame(rows, columns=self.columns, dtype="string")
        except (UnicodeError, csv.Error) as exc:
            raise DataImportError("CSV 解析失败：编码错误或引号不匹配，请另存为 UTF-8 逗号分隔 CSV。") from exc

    def read_preview(self, page: int = 1, page_size: int = 100, checkpoints: list | None = None) -> pd.DataFrame:
        """从扫描时的记录边界索引定位，支持引号内换行；只返回一页。"""
        if page < 1 or not 1 <= page_size <= 1000:
            raise ValueError("分页参数无效")
        start = (page - 1) * page_size
        checkpoint = max((p for p in (checkpoints or []) if p[0] <= start), default=None, key=lambda p: p[0])
        with self.path.open(encoding=self.encoding, newline="") as stream:
            reader = csv.reader(iter(stream.readline, ""), strict=True)
            if checkpoint:
                stream.seek(checkpoint[1])
                skip = start - checkpoint[0]
            else:
                next(reader)
                skip = start
            for _ in range(skip):
                if next(reader, None) is None:
                    return pd.DataFrame(columns=self.columns)
            rows = []
            for _ in range(page_size):
                row = next(reader, None)
                if row is None:
                    break
                rows.append(self._row(row, reader.line_num))
        return pd.DataFrame(rows, columns=self.columns, dtype="string")

    def scan_metadata(self, progress=None, cancelled=None, cache_directory=None) -> dict:
        from core.data_validator import DataQualityAnalyzer
        self.require_fields()
        return DataQualityAnalyzer().analyze(self, progress, cancelled, cache_directory)

    def import_dataset(self, service, progress=None, cancelled=None) -> dict:
        """由数据集服务统一负责缓存与事务。"""
        return service.import_dataset(self.path, progress, cancelled)
