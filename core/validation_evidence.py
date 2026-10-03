"""将测试结论绑定到当前 Python 源码，拒绝使用过期测试记录。"""
import hashlib
import json
from pathlib import Path

def source_digest(root: Path) -> str:
    hasher=hashlib.sha256()
    files=[root/"main.py"]
    for folder in ["core","app","database","tests","scripts"]:
        files.extend((root/folder).rglob("*.py"))
    for path in sorted(files):
        if path.is_file():
            hasher.update(path.relative_to(root).as_posix().encode())
            hasher.update(path.read_bytes())
    return hasher.hexdigest()

def load_test_evidence(root: Path) -> dict:
    digest=source_digest(root)
    found=False
    for name in ["stage5_2_test_results.json","stage5_1_test_results.json","stage5_test_results.json","stage4_test_results.json","stage3_test_results.json"]:
        try:
            evidence=json.loads((root/"output/reports"/name).read_text(encoding="utf-8"))
            found=True
            if evidence.get("source_digest")==digest:return evidence
        except (OSError,ValueError,KeyError):continue
    return {"status":"STALE" if found else "NOT_RUN","description":"测试记录与当前代码不一致，未据此声明通过" if found else "未找到当前版本全量测试记录"}
