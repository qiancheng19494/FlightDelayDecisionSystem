"""协调 CSV 扫描、元信息、版本缓存及源文件失效检查。"""
import json
import logging
import shutil
from pathlib import Path
from uuid import uuid4
from database.dataset_repository import DatasetRepository
from core.data_loader import FlightDataLoader, DataImportError

class DatasetService:
    def __init__(self, context):
        self.context = context
        self.repository = DatasetRepository(context.database)
        self.selected_id = None

    def loader(self, path, encoding=None):
        return FlightDataLoader(path, self.context.config.get("csv_chunksize"), encoding)

    def inspect(self, path) -> dict:
        """快速读取表头，用于正式扫描前的映射提示。"""
        loader = self.loader(path)
        info = loader.file_info()
        self.context.logger.info("字段映射：%s", info["mapping"])
        loader.require_fields()
        return info

    @staticmethod
    def source_status(record) -> str:
        path = Path(record["file_path"])
        try:
            stat = path.stat()
        except FileNotFoundError:
            return "SOURCE_MISSING"
        except OSError:
            return "SOURCE_UNREADABLE"
        if record.get("file_size") is not None and (record["file_size"] != stat.st_size or record["mtime_ns"] != stat.st_mtime_ns):
            return "SOURCE_CHANGED"
        return record["status"]

    def datasets(self) -> list[dict]:
        records = self.repository.list()
        for record in records:
            record["status"] = self.source_status(record)
        return records

    def current(self) -> dict | None:
        records = self.datasets()
        return next((r for r in records if r["id"] == self.selected_id), records[0] if records else None)

    def cached(self, record: dict) -> dict | None:
        """失效源文件或不兼容缓存不可当成当前真实统计使用。"""
        if self.source_status(record) != "READY" or not record.get("cache_path"):
            return None
        try:
            path = self.context.root / record["cache_path"]
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["quality_cache_version"] != self.context.config.get("quality_cache_version"):
                return None
            if report["file_size"] != record["file_size"] or report["mtime_ns"] != record["mtime_ns"]:
                return None
            return report
        except (OSError, ValueError, KeyError, TypeError):
            self.context.logger.warning("质量缓存无效，数据集 id=%s", record["id"])
            return None

    def _store(self, report: dict, is_demo: bool, dataset_id=None) -> dict:
        report["quality_cache_version"] = self.context.config.get("quality_cache_version")
        report["is_demo"] = is_demo
        relative = Path("data/cache") / f"dataset_{uuid4().hex}_quality.json"
        destination = self.context.root / relative
        temporary = destination.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(destination)
            identity = self.repository.save(report, relative.as_posix(), is_demo, dataset_id)
        except Exception:
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise
        return self.repository.get(identity)

    def import_dataset(self, path, progress=None, cancelled=None) -> dict:
        """先完成全部扫描才正式登记，失败不会留下成功记录。"""
        logger = self.context.logger
        logger.info("开始导入数据：%s", path)
        copied = None
        try:
            original = Path(path).resolve()
            is_demo = original.name.lower().startswith("demo_") or self.context.root / "data/demo" in original.parents
            loader = self.loader(original)
            loader.require_fields()
            logger.info("字段映射：%s", loader.mapping.to_dict())
            if self.context.config.get("copy_imported_files"):
                copied = self.context.root / self.context.config.get("data_directory") / f"{uuid4().hex}_{original.name}"
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, copied)
                loader = self.loader(copied)
            report = loader.scan_metadata(progress, cancelled, self.context.root / "data/cache")
            if cancelled and cancelled():
                raise DataImportError("导入已取消")
            record = self._store(report, is_demo)
            logger.info("导入完成：id=%s，行数=%s", record["id"], record["row_count"])
            if progress:
                progress(100, record["row_count"])
            return record
        except Exception:
            if copied:
                copied.unlink(missing_ok=True)
            logger.exception("导入失败：%s", path)
            raise

    def analyze(self, dataset_id: int, progress=None, cancelled=None) -> dict:
        """分析优先读取有效缓存；源变化后重新扫描并更新元信息。"""
        logger = self.context.logger
        logger.info("数据质量分析开始：id=%s", dataset_id)
        try:
            record = self.repository.get(dataset_id)
            report = self.cached(record)
            if report is not None:
                if progress:
                    progress(100, report["row_count"])
                logger.info("数据质量分析完成：复用缓存 id=%s", dataset_id)
                return report
            loader = self.loader(record["file_path"])
            report = loader.scan_metadata(progress, cancelled, self.context.root / "data/cache")
            updated = self._store(report, bool(record["is_demo"]), dataset_id)
            logger.info("数据质量分析完成：重新扫描 id=%s", dataset_id)
            if progress:
                progress(100, updated["row_count"])
            return report
        except Exception:
            logger.exception("数据质量分析失败：id=%s", dataset_id)
            raise

    def preview(self, dataset_id: int, page: int):
        record = self.repository.get(dataset_id)
        status = self.source_status(record)
        if status != "READY":
            raise DataImportError(f"源文件状态为 {status}，请恢复源文件或先重新分析。")
        report = self.cached(record)
        loader = self.loader(record["file_path"], record["encoding"])
        frame = loader.read_preview(page, self.context.config.get("page_size"), report.get("checkpoints") if report else None)
        if self.source_status(record) != "READY":
            raise DataImportError("预览期间源文件发生变化，请重新分析。")
        return frame

    def export_quality(self, report: dict, destination: str) -> str:
        """导出完整质量摘要 JSON，未复制源数据记录。"""
        path = Path(destination)
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        self.context.logger.info("质量报告导出完成：%s", path)
        return str(path)
