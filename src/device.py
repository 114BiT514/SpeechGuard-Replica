"""[M7 任务2] 昇腾 NPU / CUDA / CPU 设备自动选择。

`torch` 本身不认识 `npu`，必须先 `import torch_npu` 注册设备后端。
本模块把这一步收口，供 pipeline / 前端复用；在非昇腾环境（Windows/CPU）
里 import 失败会安静回退，不影响本地开发。

选择顺序：显式 preferred > npu > cuda > cpu。
"""
from __future__ import annotations

_npu_ready: bool | None = None


def register_npu() -> bool:
    """显式 import torch_npu 注册 npu 后端；不可用时返回 False。"""
    global _npu_ready
    if _npu_ready is None:
        try:
            import torch_npu  # noqa: F401
            _npu_ready = True
        except Exception:  # ImportError / 驱动缺失
            _npu_ready = False
    return _npu_ready


def pick_device(preferred: str | None = None) -> str:
    """返回可直接传给 `.to()` 的设备字符串。"""
    import torch

    if preferred:
        if preferred.startswith("npu"):
            if register_npu() and torch.npu.is_available():
                return preferred
            raise RuntimeError(
                f"请求设备 {preferred}，但 torch_npu 不可用（检查 CANN/驱动）")
        return preferred
    if register_npu() and torch.npu.is_available():
        return "npu:0"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def device_report() -> dict:
    """启动时打印/写日志用的环境摘要。"""
    import torch

    npu = register_npu() and torch.npu.is_available()
    return {
        "torch": torch.__version__,
        "npu_available": bool(npu),
        "npu_count": int(torch.npu.device_count()) if npu else 0,
        "cuda_available": torch.cuda.is_available(),
        "picked": pick_device(),
    }
