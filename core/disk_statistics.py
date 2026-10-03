"""全局重复和分位数借助临时磁盘数组，避免保存所有数据行。"""
import hashlib
import json
from pathlib import Path
import numpy as np

class DiskStatistics:
    def __init__(self, directory: Path):
        self.hash_path = directory / "row_hashes.bin"
        self.number_path = directory / "delays.bin"
        self.hash_stream = self.hash_path.open("wb")
        self.number_stream = self.number_path.open("wb")
        self.rows = self.count = 0
        self.total = 0.0
        self.minimum = self.maximum = None

    def update(self, frame, numbers) -> None:
        """按 CSV 解码后的全部原始字段计算 SHA-256，跨块识别重复。"""
        for row in frame.itertuples(index=False, name=None):
            serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.hash_stream.write(hashlib.sha256(serialized).digest())
        self.rows += len(frame)
        data = numbers.dropna().to_numpy(dtype="float64")
        if len(data):
            data.tofile(self.number_stream)
            self.count += len(data)
            self.total += float(data.sum())
            self.minimum = min(self.minimum, float(data.min())) if self.minimum is not None else float(data.min())
            self.maximum = max(self.maximum, float(data.max())) if self.maximum is not None else float(data.max())

    def close(self):
        self.hash_stream.close()
        self.number_stream.close()

    def finish(self) -> tuple[int, dict]:
        """原地磁盘映射排序/分区；分批比较，统计额外重复行数。"""
        self.close()
        duplicates = 0
        if self.rows:
            values = np.memmap(self.hash_path, dtype="S32", mode="r+", shape=(self.rows,))
            values.sort(kind="quicksort")
            for start in range(1, self.rows, 100000):
                end = min(start + 100000, self.rows)
                duplicates += int(np.count_nonzero(values[start:end] == values[start-1:end-1]))
            del values
        stats = dict(count=self.count, mean=self.total / self.count if self.count else None, min=self.minimum, max=self.maximum, median=None, p25=None, p75=None)
        if self.count:
            values = np.memmap(self.number_path, dtype="float64", mode="r+", shape=(self.count,))
            positions = [(self.count - 1) * q for q in (.25, .5, .75)]
            indices = sorted({int(np.floor(p)) for p in positions} | {int(np.ceil(p)) for p in positions})
            values.partition(indices)
            for name, p in zip(["p25", "median", "p75"], positions):
                lo, hi = int(np.floor(p)), int(np.ceil(p))
                stats[name] = float(values[lo] + (values[hi] - values[lo]) * (p - lo))
            del values
        return duplicates, stats
