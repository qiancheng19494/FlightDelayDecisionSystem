"""增量数据质量分析；报告异常但不删除源数据。"""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from core.time_parser import numeric, parse_dates, parse_hhmm
from core.delay_label import build_delay_label
from core.disk_statistics import DiskStatistics

class DataQualityAnalyzer:
    """按块统计缺失、日期、标签、类别、月度、重复与数值分布。"""
    def analyze(self, loader, progress=None, cancelled=None, cache_directory=None) -> dict:
        loader.require_fields()
        info = loader.file_info()
        missing = Counter({c: 0 for c in loader.columns})
        types = {c: set() for c in loader.columns}
        counts = Counter()
        issues = Counter()
        categories = {c: set() for c in ["airline", "origin", "destination"]}
        monthly = {}
        start_date = end_date = None
        warnings = []
        if "cancelled" not in loader.mapping.mapped_fields.values():
            warnings.append("缺少取消状态字段：暂按未取消统计标签，无法核实取消航班；使用标签前需确认。")
        if cache_directory:
            Path(cache_directory).mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="scan_", dir=cache_directory) as temporary:
            disk = DiskStatistics(Path(temporary))
            try:
                for raw in loader.iter_chunks(progress, cancelled):
                    if cancelled and cancelled():
                        raise ValueError("任务已取消")
                    # 空白字符串为缺失；NA 航司代码不当成缺失，避免 pandas 默认 NA 猜测。
                    clean = raw.apply(lambda col: col.str.strip().replace("", pd.NA))
                    frame = clean.rename(columns=loader.mapping.mapped_fields)
                    counts["row_count"] += len(raw)
                    counts["empty_rows"] += int(clean.isna().all(axis=1).sum())
                    for col in raw.columns:
                        missing[col] += int(clean[col].isna().sum())
                        nonempty = clean[col].dropna()
                        if len(nonempty):
                            nums = numeric(nonempty)
                            if nums.notna().any():
                                types[col].add("number")
                            if nums.isna().any():
                                types[col].add("text")
                    dates = parse_dates(frame["flight_date"])
                    counts["valid_dates"] += int(dates.notna().sum())
                    issues["无效或缺失日期"] += int(dates.isna().sum())
                    if dates.notna().any():
                        lo, hi = dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")
                        start_date = min(start_date, lo) if start_date else lo
                        end_date = max(end_date, hi) if end_date else hi
                    for name in categories:
                        categories[name].update(frame[name].dropna().unique().tolist())
                        issues[{"airline": "航司缺失", "origin": "始发机场缺失", "destination": "目的机场缺失"}[name]] += int(frame[name].isna().sum())
                    issues["始发与目的机场相同"] += int((frame.origin.notna() & frame.origin.eq(frame.destination)).sum())
                    for name, condition, title in [("distance", lambda n: n < 0, "距离为负"), ("scheduled_elapsed_time", lambda n: n <= 0, "计划时长非正")]:
                        if name in frame:
                            nums = numeric(frame[name])
                            issues[title] += int(condition(nums).sum())
                            issues[name + " 非法数值"] += int((frame[name].notna() & nums.isna()).sum())
                    for name in ["scheduled_departure", "scheduled_arrival"]:
                        if name in frame:
                            # 按块内唯一值解析，避免对百万行重复调用 Python 解析器。
                            parsed = {value: parse_hhmm(value) for value in frame[name].dropna().unique()}
                            issues[name + " 无效时间"] += int(frame[name].map(lambda v: False if pd.isna(v) else parsed[v].text is None).sum())
                            counts[name + "_next_day"] += int(frame[name].map(lambda v: False if pd.isna(v) else parsed[v].day_offset == 1).sum())
                    cancellation = frame.get("cancelled")
                    if cancellation is not None:
                        flags = numeric(cancellation)
                        counts["cancelled_count"] += int(flags.eq(1).sum())
                        issues["取消状态无效或缺失"] += int((~flags.isin([0, 1])).sum())
                    delays = numeric(frame["departure_delay"]) if "departure_delay" in frame else pd.Series(dtype=float)
                    labels = pd.Series(pd.NA, index=frame.index, dtype="Int64")
                    if "departure_delay" in frame:
                        labels = build_delay_label(frame.departure_delay, cancellation)
                        issues["延误值无法转为有限数值"] += int((frame.departure_delay.notna() & delays.isna()).sum())
                    counts["valid_labeled_count"] += int(labels.notna().sum())
                    counts["delay_count"] += int(labels.eq(1).sum())
                    counts["normal_count"] += int(labels.eq(0).sum())
                    for month, subset in pd.DataFrame({"month": dates.dt.strftime("%Y-%m"), "label": labels}).dropna(subset=["month"]).groupby("month"):
                        record = monthly.setdefault(month, {"rows": 0, "labeled": 0, "delayed": 0})
                        record["rows"] += len(subset)
                        record["labeled"] += int(subset.label.notna().sum())
                        record["delayed"] += int(subset.label.eq(1).sum())
                    disk.update(raw, delays)
                if progress:
                    progress(94, counts["row_count"])
                duplicates, delay_stats = disk.finish()
            finally:
                disk.close()
        if cancelled and cancelled():
            raise ValueError("任务已取消")
        if info["file_size"] != loader.path.stat().st_size or info["mtime_ns"] != loader.path.stat().st_mtime_ns:
            raise ValueError("扫描期间源文件发生变化，请停止编辑后重新导入。")
        rows, labeled = counts["row_count"], counts["valid_labeled_count"]
        if not rows:
            warnings.append("CSV 只有表头，没有数据记录。")
        if rows and not counts["valid_dates"]:
            warnings.append("所有日期均无效，日期范围和月度图不可计算。")
        if "departure_delay" in loader.mapping.mapped_fields.values() and not labeled:
            warnings.append("当前数据集没有可用真实延误标签。")
        fields = [{"field": c, "standard_field": loader.mapping.mapped_fields.get(c, "未识别"), "dtype": ("mixed" if len(types[c]) > 1 else next(iter(types[c]), "empty")), "non_null": rows-missing[c], "missing": missing[c], "missing_rate": missing[c]/rows if rows else 0} for c in loader.columns]
        return {**info, **{k: counts[k] for k in ["row_count", "empty_rows", "valid_dates", "cancelled_count", "valid_labeled_count", "delay_count", "normal_count"]}, "column_count": len(loader.columns), "invalid_dates": issues["无效或缺失日期"], "start_date": start_date, "end_date": end_date, "dataset_type": loader.mapping.dataset_type.value, "airline_count": len(categories["airline"]), "origin_count": len(categories["origin"]), "destination_count": len(categories["destination"]), "delay_rate": counts["delay_count"]/labeled if labeled else None, "cancelled_rate": counts["cancelled_count"]/rows if rows and "cancelled" in loader.mapping.mapped_fields.values() else None, "duplicate_rows": duplicates, "duplicate_method": "SHA-256 of decoded raw CSV fields; extra occurrences", "delay_statistics": delay_stats, "fields": fields, "issues": dict(issues), "warnings": warnings, "monthly_summary": dict(sorted(monthly.items())), "checkpoints": loader.checkpoints, "time_2400_count": sum(counts[c+"_next_day"] for c in ["scheduled_departure", "scheduled_arrival"]), "missing_required_fields": loader.mapping.missing_required_fields}
