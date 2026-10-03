"""JSON 配置读取、校验与原子写入。"""
import json
from pathlib import Path

class ConfigManager:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.path = self.root / "config/user_config.json"
        self.warnings = []
        self.values = json.loads((self.root / "config/default_config.json").read_text(encoding="utf-8"))
        self.validate(self.values)
        if self.path.exists():
            try:
                user = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(user, dict):
                    raise ValueError("配置顶层必须为对象")
                candidate = {**self.values, **user}
                self.validate(candidate)
                self.values = candidate
            except (OSError, ValueError, TypeError) as exc:
                self.warnings.append(f"用户配置无效，已使用默认配置，原文件已保留：{exc}")

    @staticmethod
    def validate(values):
        """拒绝无效阈值、目录和界面参数，防止启动阶段出错。"""
        if type(values.get("csv_chunksize", 100000)) is not int or not 100 <= values.get("csv_chunksize", 100000) <= 1000000:
            raise ValueError("CSV 分块大小必须为 100 至 1000000")
        if type(values.get("copy_imported_files", False)) is not bool:
            raise ValueError("copy_imported_files 必须为布尔值")
        if values.get("quality_cache_version", 1) != 1:
            raise ValueError("不支持的质量缓存版本")
        from core.history_schema import history_config
        history_config(values.get("historical_features"))
        threshold = values["default_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("默认阈值必须在 0 到 1 之间")
        for key, lower, upper in [("page_size", 10, 1000), ("window_width", 1000, 3840), ("window_height", 640, 2160)]:
            if type(values[key]) is not int or not lower <= values[key] <= upper:
                raise ValueError(f"{key} 超出允许范围")
        for key in ["data_directory", "export_directory"]:
            value = values[key]
            if not isinstance(value, str) or not value.strip() or Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError(f"{key} 必须为项目内相对路径")
        if values["log_level"] not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError("不支持的日志级别")
        if values["default_model"] not in ["Random Forest", "XGBoost"] or values["theme"] != "light":
            raise ValueError("不支持的默认模型或主题")
        if values["version"] != "V1.0":
            raise ValueError("配置版本不兼容")

    def save(self, updates):
        """校验成功后原子替换配置，不破坏已有有效文件。"""
        candidate = {**self.values, **updates}
        self.validate(candidate)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
        self.values = candidate

    def get(self, key):
        return self.values[key]
