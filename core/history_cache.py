"""独立于质量缓存的历史缓存；原子指针只指向已构建且已审计的版本。"""
import hashlib
import json
from pathlib import Path
from core.history_schema import ENGINE_VERSION, SCHEMA_VERSION

def file_fingerprint(path: Path) -> dict:
    stat=path.stat()
    return {"source_file":str(path.resolve()),"source_file_size":stat.st_size,"source_modified_time":stat.st_mtime_ns}

def digest(path: Path) -> str:
    hasher=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            hasher.update(block)
    return hasher.hexdigest()

class HistoryCache:
    def __init__(self, root: Path, dataset_id: int):
        self.directory=root/"data/cache/historical"/f"dataset_{int(dataset_id)}"
        self.pointer=self.directory/"current.json"

    def load(self, source: Path, config: dict, verify_files=True):
        try:
            pointer=json.loads(self.pointer.read_text(encoding="utf-8"))
            folder=self.directory/pointer["generation"]
            if folder.parent.resolve()!=self.directory.resolve():
                return None,"缓存路径不合法"
            metadata=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
            if metadata["engine_version"]!=ENGINE_VERSION or metadata["historical_feature_schema_version"]!=SCHEMA_VERSION:
                return None,"引擎或特征版本变化"
            if metadata["configuration"]!=config:
                return None,"历史配置变化"
            if any(metadata[k]!=v for k,v in file_fingerprint(source).items()):
                return None,"源文件大小、路径或修改时间变化"
            if verify_files:
                for name,sha in metadata["table_sha256"].items():
                    if digest(folder/name)!=sha:
                        return None,"历史统计文件损坏"
            return (folder,metadata),"有效历史缓存"
        except (OSError,ValueError,KeyError,TypeError):
            return None,"暂无可用历史缓存，或缓存/源文件不存在"

    def publish(self, generation: Path, metadata: dict):
        (generation/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        temporary=self.directory/(generation.name+".pointer.tmp")
        temporary.write_text(json.dumps({"generation":generation.name}),encoding="utf-8")
        temporary.replace(self.pointer)
