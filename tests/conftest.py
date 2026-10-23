"""让 tests/ 能直接 import src 包（无需安装，pytest 从仓库根运行）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
