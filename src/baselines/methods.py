"""[M5 任务5] 对比基线：单参数分段线性（公式(2)）、McAdams、Overo。

论文 §4.2.3 / Fig.10-11 的对比对象（全部是频域扭曲家族，SpeechGuard 的差异在于
"多参数 + 可逆 + 抗 reducing 攻击"）：

- single（公式(2)）：单参数两段线性，折点 ω0=7π/8（α≤1）或 7π/(8α)（α>1），
  已有 piecewise_warp；逆变换用插值表（单调）。
- mcadams：McAdams et al.1998——正弦频谱移位畸变（pitch 域变换）。此处按论文
  "系数变换"描述实现为线性频率轴上的正弦 warp：y = x + A·sin(2π(x−c)/P)，
  约束 A·2π/P < 1 保证单调可逆（参数档案：A,c,P per segment）。
- overo：双参数分段线性（α,β 独立均匀抽样，论文引 Overo[35]），即 multi_warp
  n=1 但**不加斜率约束**（这正是与 SpeechGuard 多参数版的对照点）。

所有基线的 protect/recover 与 src.acoustic.warping 同一帧化/网格/DFT 管线
（块级 single 模式除外，这里逐段映射 + OLA 用 single 整句映射以对齐各自方法学）。
"""
from __future__ import annotations

import numpy as np

from src.acoustic.warping import (FINE_N, FRAME_LEN, SegmentParams, _warp_frames,
                                  piecewise_warp, multi_warp, PI)
from src.preprocess.framing import make_frames
from src.preprocess.vad import merge_short_segments, segmentation, voiced_flags
from src.preprocess.stft import frames_centered, istft, sqrt_hann

MIN_SEG_FRAMES = 3


# ───────────── 映射构造：返回 (fwd(omega), inv(omega)) 函数对 ─────────────

def _table_pair(fwd: np.ndarray, grid: np.ndarray) -> tuple:
    """单调 fwd 网格 → (正向插值, 反向插值) 函数对。"""
    def f(w):
        return np.interp(w, grid, fwd)

    def finv(w):
        return np.interp(w, fwd, grid)
    return f, finv


def map_single(alpha: float):
    grid = np.linspace(0.0, PI, 4001)
    return _table_pair(piecewise_warp(grid, alpha), grid)


def map_mcadams(A: float, c: float, P: float):
    assert A * 2 * PI / P < 1.0, "McAdams 参数不满足单调可逆条件"
    grid = np.linspace(0.0, PI, 4001)
    fwd = grid + A * np.sin(2 * PI * (grid - c) / P)
    fwd[0], fwd[-1] = 0.0, PI                      # 边界钉住 [0,π]
    return _table_pair(fwd, grid)


def map_overo(alpha: np.ndarray, beta: np.ndarray):
    grid = np.linspace(0.0, PI, 4001)
    return _table_pair(multi_warp(grid, alpha, beta), grid)


def _clip_open(vals, rng, margin=0.05):
    v = np.asarray(vals, np.float64)
    return np.clip(v, margin * PI, (1 - margin) * PI)


# ───────────── 参数抽样 ─────────────

def gen_params_single(rng):
    """公式(2)：α ∈ [0.6, 1.7]（论文图对比用的典型音高/频谱形变范围）。"""
    return float(rng.uniform(0.6, 1.7))


def gen_params_mcadams(rng):
    A = rng.uniform(0.05, 0.25) * PI
    P = rng.uniform(0.6, 1.2) * PI
    c = rng.uniform(0.15, 0.85) * PI
    while A * 2 * PI / P >= 0.98:                  # 单调约束
        A *= 0.9
    return (float(A), float(c), float(P))


def gen_params_overo(rng):
    """Overo 双参数：α,β 独立均匀（无斜率约束），各 1 个内点。"""
    a = rng.uniform(0.1, 0.9) * PI
    b = rng.uniform(0.1, 0.9) * PI
    return (np.array([a]), np.array([b]))


# ───────────── 通用整句管线（single 式：整句一个映射 + OLA 合成） ─────────────

def _apply_maps(wav, seg_maps, invert):
    wav = np.asarray(wav, dtype=np.float64).ravel()
    frames, _ = frames_centered(wav)
    win = frames * sqrt_hann(FRAME_LEN)
    fine = np.fft.rfft(win, n=FINE_N, axis=-1)
    omega_fine = np.linspace(0.0, PI, FINE_N // 2 + 1)
    omega = np.linspace(0.0, PI, FRAME_LEN // 2 + 1)
    new_spec = np.empty((frames.shape[0], omega.size), dtype=np.complex128)
    for (i0, i1, f, finv) in seg_maps:
        tgt = (finv if invert else f)(omega)
        idx = np.clip(np.searchsorted(omega_fine, tgt, side="right") - 1,
                      0, len(omega_fine) - 2)
        denom = omega_fine[idx + 1] - omega_fine[idx]
        w1 = (tgt - omega_fine[idx]) / denom
        S = np.zeros((len(tgt), len(omega_fine)))
        S[np.arange(len(tgt)), idx] = 1.0 - w1
        S[np.arange(len(tgt)), idx + 1] = w1
        new_spec[i0:i1] = fine[i0:i1] @ S.T
    return istft(new_spec, target_len=wav.size)


def _segments(wav):
    """整句一段（与 SpeechGuard single 模式同框架，公平对比）。"""
    n_frames = int(np.ceil(len(np.asarray(wav).ravel()) / FRAME_LEN))
    return [(0, n_frames)]


METHODS = {
    "single":  ("single2",  gen_params_single),
    "mcadams": ("mcadams",  gen_params_mcadams),
    "overo":   ("overo",    gen_params_overo),
}


def _build_map(method, p):
    if method == "single":
        return map_single(p)
    if method == "mcadams":
        return map_mcadams(*p)
    if method == "overo":
        return map_overo(*p)
    raise KeyError(method)


def protect(wav, method: str, rng=None):
    """返回 (protected_wav, archive)。archive=[params]（整句一段，与
    SpeechGuard single 模式同框架以保证对比公平；params 结构见 gen_params_*）。"""
    rng = np.random.default_rng() if rng is None else rng
    _, gen = METHODS[method]
    wav = np.asarray(wav, dtype=np.float64).ravel()
    params = gen(rng)
    f, finv = _build_map(method, params)
    segs = [(0, frames_centered(wav)[0].shape[0], f, finv)]
    return _apply_maps(wav, segs, invert=False), [params]


def recover(wav_protected, archive, method: str):
    wav_protected = np.asarray(wav_protected, dtype=np.float64).ravel()
    (params,) = archive
    f, finv = _build_map(method, params)
    segs = [(0, frames_centered(wav_protected)[0].shape[0], f, finv)]
    return _apply_maps(wav_protected, segs, invert=True)


def gen_attack_params(method, rng):
    """reducing 攻击用：攻击者已知方法与其参数空间，随机猜参数做逆扭曲。"""
    _, gen = METHODS[method]
    return gen(rng)
