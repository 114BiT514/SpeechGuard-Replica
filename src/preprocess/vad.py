"""[M2 任务2] 动态阈值 VAD（论文 §4.1 公式(1)）。

公式(1)：Φ(k) = at·( max(|X_k(e^{jω})|) + δ ) + Σ_{i=1..T} a_i·Φ(k−i)
论文取值：T=1, at=0.7, a_1=0.25, δ=1.99, Φ(−1) = max(|X_0|) + δ。
分类：max(|X_k|) < Φ(k) → 无声帧；max(|X_k|) ≥ Φ(k) → 有声帧。

连续同类帧合并为段（segment），段是 warping 参数分配的基本单位
（论文 §4.2.2：共 v 个段，每段独立一组 2n 个参数）。
"""
from __future__ import annotations

import numpy as np

# 论文 §4.1 给定的 VAD 参数
AT = 0.7        # 当前帧峰值权重
A1 = 0.25       # 一阶预测系数（T=1）
DELTA = 1.99    # 噪底保护间隔


def voiced_flags(spec, at: float = AT, a1: float = A1, delta: float = DELTA) -> np.ndarray:
    """逐帧频谱 → (n_frames,) 布尔数组（True=有声帧）。

    spec 为 stft 输出 (n_frames, bins)。递归阈值按公式(1)逐帧更新。
    """
    spec = np.asarray(spec)
    peaks = np.abs(spec).max(axis=1)
    phi = np.empty_like(peaks)
    prev = peaks[0] + delta  # Φ(−1)，论文给定
    for k, mk in enumerate(peaks):
        phi[k] = at * (mk + delta) + a1 * prev
        prev = phi[k]
    return peaks >= phi


def segmentation(flags: np.ndarray) -> list[tuple[int, int, bool]]:
    """逐帧布尔标记 → 段列表 [(fs, fe, voiced)]。

    fs/fe 为闭区间帧号；相邻同类帧合并，全序列恰好被段覆盖且互不重叠。
    """
    flags = np.asarray(flags, dtype=bool)
    n = flags.size
    segs: list[tuple[int, int, bool]] = []
    start = 0
    for i in range(1, n + 1):
        if i == n or flags[i] != flags[start]:
            segs.append((start, i - 1, bool(flags[start])))
            start = i
    return segs


def merge_short_segments(segs: list[tuple[int, int, bool]],
                         min_len: int = 1) -> list[tuple[int, int, bool]]:
    """把短于 min_len 的段并入邻段（工程平滑，论文未提及；min_len=1 时原样返回）。

    合并规则：短段优先并入前一段，位于开头则并入后一段；合并后的 voiced 标记
    取"任一组成部分有声即为有声"。合并只影响 warping 的参数分配粒度。
    """
    if min_len <= 1 or not segs:
        return list(segs)
    out = [list(segs[0])]
    for seg in segs[1:]:
        fs, fe, voiced = seg
        if fe - fs + 1 < min_len:
            if len(out) > 0:
                out[-1][1] = fe                      # 并入前一段
                out[-1][2] = out[-1][2] or voiced
            else:
                out.append([fs, fe, voiced])
        else:
            if out and out[-1][1] + 1 == fs and out[-1][2] == voiced:
                out[-1][1] = fe                       # 相邻同类段继续合并
            else:
                out.append([fs, fe, voiced])
    # 收尾：若第一段仍过短（开头短段无前段可并），由第二段吸收
    if len(out) > 1 and out[0][1] - out[0][0] + 1 < min_len:
        out[1][0] = out[0][0]
        out[1][2] = out[1][2] or out[0][2]
        out.pop(0)
    return [tuple(s) for s in out]
