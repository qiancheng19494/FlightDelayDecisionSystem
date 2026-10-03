"""独立执行数据库初始化，不要求 GUI 依赖。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.bootstrap import initialize
if __name__ == "__main__":
    context = initialize(ROOT)
    print(f"数据库初始化完成：{context.database.path}")
