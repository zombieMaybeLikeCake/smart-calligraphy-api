import sys
from pathlib import Path

# 讓 `from app.main import app` 在從任何目錄跑 pytest 時都能 import 到，
# 不依賴 pytest 的 rootdir 自動插入行為。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
