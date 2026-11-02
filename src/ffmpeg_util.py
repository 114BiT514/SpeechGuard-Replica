"""[M7] ffmpeg 定位：MP3 发布链路要求 ffmpeg 自带 libmp3lame 编码器。

云端 ModelArts 镜像自带的 `/usr/local/ffmpeg/bin/ffmpeg`（4.2）未编入
libmp3lame，直接调用会在 encode_mp3 报 "Default encoder for format mp3 ...
disabled"。查找顺序：

  SG_FFMPEG 环境变量 → PATH 上的 ffmpeg（须含 libmp3lame）→ imageio-ffmpeg 自带静态版

imageio-ffmpeg 的 aarch64 静态包含 libmp3lame，可用
`pip install imageio-ffmpeg` 兜底。结果做进程内缓存。
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess

_LEGACY_WIN = r"E:/Miniconda/envs/speechguard/Library/bin/ffmpeg.exe"


@functools.lru_cache(maxsize=4)
def _has_lame(ff: str) -> bool:
    try:
        out = subprocess.run([ff, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=20).stdout
        return "libmp3lame" in out
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def find_ffmpeg() -> str:
    env = os.environ.get("SG_FFMPEG")
    if env and os.path.exists(env):
        return env
    sys_ff = shutil.which("ffmpeg")
    if sys_ff and _has_lame(sys_ff):
        return sys_ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    if sys_ff:
        return sys_ff
    if os.path.exists(_LEGACY_WIN):
        return _LEGACY_WIN
    raise FileNotFoundError(
        "找不到可用 ffmpeg；请安装含 libmp3lame 的版本并设置 SG_FFMPEG")
