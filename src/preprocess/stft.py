"""[M2 任务3] STFT/ISTFT 封装：窗长对齐 36ms 帧，sqrt-Hann + 窗和归一化（OLA 完美重构）。

论文只规定"对每帧做 STFT"（§4.1），未指定窗与重叠。这里的选择（报告将注明为实现自由度）：
- 分析窗 = 合成窗 = sqrt(周期 Hann)，hop = frame_len//2（50% 重叠，288 采样点）；
- 信号首端补 frame_len//2 点（centering），使真实首样本落在窗的峰值附近；
- 合成端除以窗和（window-sum-square），在窗和 > 0 处完美重构：istft(stft(x)) ≈ x。

完美重构是 warping"保护→恢复"闭环可逆的前提：频域扭曲在帧谱上进行，
时域重建必须不引入额外损失，恢复误差才只来自扭曲/逆扭曲本身的插值。
"""
from __future__ import annotations

import numpy as np

from .framing import FRAME_LEN, make_frames

DEFAULT_HOP = FRAME_LEN // 2  # 288，50% 重叠
CENTER_PAD = FRAME_LEN // 2   # 首端补零，配合 50% 重叠保证所有真实样本窗和充足


def sqrt_hann(n: int) -> np.ndarray:
    """周期 Hann 窗的平方根（w[0]=0，中点峰值 1，周期性结尾）。"""
    i = np.arange(n)
    return np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * i / n))


def frames_centered(wav, frame_len: int = FRAME_LEN, hop: int | None = None):
    """波形 → (带 centering 补零的帧矩阵, pad)。stft 内部与 warping 模块共用，
    保证各模块看到的帧网格一致。"""
    hop = DEFAULT_HOP if hop is None else hop
    x = np.asarray(wav, dtype=np.float64).ravel()
    x = np.concatenate([np.zeros(CENTER_PAD), x])
    return make_frames(x, frame_len, hop)


def stft(wav, frame_len: int = FRAME_LEN, hop: int | None = None, window: np.ndarray | None = None):
    """波形 → (逐帧频谱, pad)。

    频谱形状 (n_frames, frame_len//2 + 1)，complex128；pad 为 make_frames 的末尾补零数。
    centering 补零在本函数内部完成，istft 端对称地去掉。
    """
    hop = DEFAULT_HOP if hop is None else hop
    w = sqrt_hann(frame_len) if window is None else np.asarray(window, dtype=np.float64)
    frames, pad = frames_centered(wav, frame_len, hop)
    spec = np.fft.rfft(frames * w, n=frame_len, axis=-1)
    return spec, pad


def istft(spec, frame_len: int = FRAME_LEN, hop: int | None = None,
          window: np.ndarray | None = None, target_len: int | None = None) -> np.ndarray:
    """逐帧频谱 → 波形（重叠相加 + 窗和归一化，再去掉 centering 补零）。

    target_len 给定时裁剪回原长；不给定则返回去掉 CENTER_PAD 后的全部有效长度。
    """
    hop = DEFAULT_HOP if hop is None else hop
    w = sqrt_hann(frame_len) if window is None else np.asarray(window, dtype=np.float64)
    frames = np.fft.irfft(spec, n=frame_len, axis=-1) * w
    n_frames = spec.shape[0]
    out_len = (n_frames - 1) * hop + frame_len
    y = np.zeros(out_len)
    wss = np.zeros(out_len)
    for i in range(n_frames):
        s = i * hop
        y[s:s + frame_len] += frames[i]
        wss[s:s + frame_len] += w * w
    mask = wss > 1e-12
    y[mask] /= wss[mask]
    y[~mask] = 0.0
    y = y[CENTER_PAD:]
    if target_len is not None:
        y = y[:target_len]
    return y
