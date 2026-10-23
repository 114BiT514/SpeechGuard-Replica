"""[M2 任务4] 多参数分段线性频域扭曲（warping）——全项目技术核心。

论文 §4.2：
- 公式(2)：单参数两段线性扭曲（对比基线）。ω0 = 7π/8（α≤1）或 7π/(8α)（α>1）；
- 公式(3)：多参数扭曲 m(ω,α,β)：穿过点列 (0,0),(α1,β1),…,(αn,βn),(π,π) 的分段线性
  函数，α、β 严格递增且在 (0,π) 内（边界 α0=β0=0, α_{n+1}=β_{n+1}=π）→ m 严格
  递增 → [0,π]→[0,π] 的双射；
- 逆函数：m⁻¹(ω,α,β) = m(ω,β,α)（α、β 互换），这是"可恢复"的全部来源；
- 公式(4)：扭曲强度 dist = ∫₀^π |F(ω)−ω| dω；论文 §5.1 限定 dist∈(0.5,0.6)、n∈[1,10]。

实现说明与已知局限（报告将展开，亦是加分项素材）：
- 采用**无重叠矩形块**处理（hop = 36ms，与论文公式(5)的帧语义一致）：
  块 → DFT → 逐段扭曲 → IDFT → 块。恢复与保护共用同一管线，仅把 m 换成 m⁻¹。
- **精细 DTFT 网格必须整除覆盖 DFT 网格**：FINE_N = 16×FRAME_LEN。若不成整数倍
  （如 8192/576=14.22），恒等映射也要跨网格缝隙线性插值，roundtrip 直接崩坏
  （实测恒等仅 13dB；修复后恒等 ≈300dB）。
- **离散化不可逆（固有损失，非 bug）**：论文 §4.2.2 的逆在连续意义下精确，但离散
  实现中（a）分片线性映射在 knots 处的斜率间断与（b）矩形窗谱峰的巨大导数相乘，
  使映射后谱的傅里叶系数衰减变慢，有限块 DFT 表示产生 O(1) 截断误差；表现为
  roundtrip SNR 随参数抽样波动（实测 median≈6dB，min≈1dB，max≈20dB）。
  论文 §5.2 自己承认"actual implementation operations such as interpolation
  result in differences from the original results"，且其恢复质量为 EER 7.07% /
  WER 9.90%（原始 WER 2.46%）——恢复有损与本文实现一致。M3 将以恢复后
  EER/WER（而非波形 SNR）作为复现对齐指标；M7 加分项可探索改进恢复管线。
- **斜率约束**：论文只约束 α、β 严格递增与 dist 范围；纯随机抽样会以可观概率产生
  数十倍局部斜率。此处约束分段斜率 ∈ [0.5, 2]（VTLN 常规范围），报告中注明。
- **参数构造**：Dirichlet 切段 + 斜率偏差加权中心化（保证终点约束与斜率界限严格
  成立），仅对 dist 做拒绝采样；独立排序抽样在斜率受限时对 n≥8 几乎不可行。
- 每个 VAD 段（含无声段）独立一组参数，段内所有帧共用；过短段先合并。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.preprocess.framing import FRAME_LEN, make_frames
from src.preprocess.vad import merge_short_segments, segmentation, voiced_flags

PI = np.pi
FINE_FACTOR = 16                  # 精细 DTFT 过采样倍数（必须为整数倍！）
FINE_N = FRAME_LEN * FINE_FACTOR  # 9216：精细网格必须整除覆盖 289 点 DFT 网格，
# 否则恒等映射也要跨缝隙插值，roundtrip 直接崩坏（实测 13dB 的教训）
DIST_RANGE = (0.5, 0.6)           # 论文 §5.1
N_RANGE = (1, 10)                 # 论文 §5.1
MIN_SEG_FRAMES = 3                # 工程平滑：短于此帧数的 VAD 段并入邻段
SLOPE_MAX = 2.0                   # 工程约束：分段斜率 ∈ [1/SLOPE_MAX, SLOPE_MAX]。
# 论文只约束 α、β 严格递增与 dist 范围，未约束斜率；纯随机抽样会以可观概率产生
# 数十倍局部斜率（频谱剧烈拉伸/压缩），报告中作为实现约束注明。


@dataclass
class SegmentParams:
    """一个段的扭曲参数档案（恢复密钥的组成部分）。"""

    frame_start: int
    frame_end: int      # 闭区间
    alpha: np.ndarray   # (n,)
    beta: np.ndarray    # (n,)
    dist: float

    @property
    def n(self) -> int:
        return len(self.alpha)


# ---------------------------------------------------------------- 公式(2)(3)(4)

def piecewise_warp(omega, alpha: float) -> np.ndarray:
    """公式(2)：单参数两段线性扭曲（基线方法）。α>1 频谱上移（变尖），α<1 下移（变低沉）。"""
    omega = np.asarray(omega, dtype=np.float64)
    w0 = 7 * PI / 8 if alpha <= 1 else 7 * PI / (8 * alpha)
    up = alpha * omega
    down = alpha * w0 + (PI - alpha * w0) / (PI - w0) * (omega - w0)
    return np.where(omega <= w0, up, down)


def multi_warp(omega, alpha, beta) -> np.ndarray:
    """公式(3)：穿过点列 (0,0),(α_i,β_i),…,(π,π) 的分段线性函数。"""
    alpha = np.concatenate([[0.0], np.asarray(alpha, dtype=np.float64), [PI]])
    beta = np.concatenate([[0.0], np.asarray(beta, dtype=np.float64), [PI]])
    omega = np.asarray(omega, dtype=np.float64)
    seg = np.clip(np.searchsorted(alpha, omega, side="right") - 1, 0, alpha.size - 2)
    slope = (beta[seg + 1] - beta[seg]) / (alpha[seg + 1] - alpha[seg])
    return slope * (omega - alpha[seg]) + beta[seg]


def inverse_multi_warp(omega, alpha, beta) -> np.ndarray:
    """逆函数（论文 §4.2.2）：m⁻¹(ω,α,β) = m(ω,β,α)。"""
    return multi_warp(omega, beta, alpha)


def distortion_strength(alpha=None, beta=None, warp=None, n_grid: int = 20001) -> float:
    """公式(4)：dist = ∫₀^π |F(ω)−ω| dω（梯形数值积分）。identity 时为 0。"""
    grid = np.linspace(0.0, PI, n_grid)
    f = warp(grid) if warp is not None else multi_warp(grid, alpha, beta)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(np.abs(f - grid), grid))


def gen_params(n: int | None = None, dist_range=DIST_RANGE,
               rng: np.random.Generator | None = None, max_trials: int = 20000):
    """随机生成 n 对参数：α、β ∈ (0,π) 严格递增，dist 落入 dist_range。

    n=None 时按论文 §5.1 在 [1,10] 内均匀随机。

    构造法（而非独立排序抽样）：先把 [0,π] 按 Dirichlet 随机切成 n+1 段得到 α；
    斜率取 1+u（u∈[-0.5,1]，即斜率∈[0.5, SLOPE_MAX]），对 u 做以段宽为权的
    中心化使 Σ 段宽·斜率 = π（终点约束自动成立，β 由斜率累加得到）；
    最后仅对 dist 做拒绝采样。独立排序抽样在斜率受限时对 n≥8 几乎不可行，
    构造法保证任意 n 都可行。返回 (alpha, beta, dist)。
    """
    rng = np.random.default_rng() if rng is None else rng
    if n is None:
        n = int(rng.integers(N_RANGE[0], N_RANGE[1] + 1))
    lo, hi = dist_range
    k = n + 1  # 段数（含首尾边界段）
    slo_lo, slo_hi = 1.0 / SLOPE_MAX, SLOPE_MAX
    for _ in range(max_trials):
        gaps = rng.dirichlet(np.ones(k)) * PI
        u = rng.uniform(slo_lo - 1.0, slo_hi - 1.0, k)
        u -= float((gaps * u).sum() / PI)          # 加权中心化 → 终点约束
        # 迭代吸收残差：clip 后中心化约束可能被破坏，把残差交给余量最大的元素，
        # 直到 Σ 段宽·斜率 = π 且所有斜率严格在界内（k≤11，通常 2~3 轮收敛）
        for _ in range(20):
            np.clip(u, slo_lo - 1.0, slo_hi - 1.0, out=u)
            r = float((gaps * u).sum() / PI)
            if abs(r) < 1e-12:
                break
            idx = int(np.argmax(u)) if r > 0 else int(np.argmin(u))
            take = max(min(r, slo_hi - 1.0 - u[idx]), min(r, slo_lo - 1.0 - u[idx]))
            u[idx] -= take
        slopes = 1.0 + u
        alpha = np.cumsum(gaps)[:-1]
        beta = np.cumsum(slopes * gaps)[:-1]
        if np.any(alpha <= 0) or np.any(alpha >= PI) or np.any(beta <= 0) or np.any(beta >= PI):
            continue
        d = distortion_strength(alpha, beta)
        if lo <= d <= hi:
            return alpha, beta, d
    raise RuntimeError(f"gen_params: {max_trials} 次内未抽到满足 dist∈{dist_range} 且"
                       f"斜率∈[{slo_lo:.2g},{slo_hi:.2g}] 的参数（n={n}）")


# ---------------------------------------------------------------- protect / recover

def _frame_view(wav):
    """无重叠 36ms 帧化 + 逐帧矩形 DFT。返回 (frames, spec, omega_grid)。"""
    frames, _pad = make_frames(wav)                      # hop = FRAME_LEN，无重叠
    spec = np.fft.rfft(frames, axis=-1)
    omega = np.linspace(0.0, PI, spec.shape[1])
    return frames, spec, omega


def _warp_frames(frames: np.ndarray, omega: np.ndarray, freq_map) -> np.ndarray:
    """对帧组做频域扭曲：Y[k] = X(m(ω_k))。

    frames: (k, FRAME_LEN) 时域块（无窗）。X 先在 FINE_N 点精细网格上精确采样，
    再线性插值到 m(ω_k)。返回与 irfft(·, n=FRAME_LEN) 兼容的复数谱。
    """
    fine = np.fft.rfft(frames, n=FINE_N, axis=-1)
    omega_fine = np.linspace(0.0, PI, FINE_N // 2 + 1)
    target = np.asarray(freq_map(omega), dtype=np.float64)
    out = np.empty((frames.shape[0], omega.size), dtype=np.complex128)
    for i in range(frames.shape[0]):
        out[i].real = np.interp(target, omega_fine, fine[i].real)
        out[i].imag = np.interp(target, omega_fine, fine[i].imag)
    return out


def _apply(wav, archive=None, n=None, dist_range=DIST_RANGE, rng=None, invert=False):
    """protect/recover 共用管线：invert=False 扭曲（保护），True 逆扭曲（恢复）。"""
    wav = np.asarray(wav, dtype=np.float64).ravel()
    frames, spec, omega = _frame_view(wav)
    new_spec = spec.copy()

    if not invert:
        flags = voiced_flags(spec)
        segs = merge_short_segments(segmentation(flags), MIN_SEG_FRAMES)
        archive = []
        for fs, fe, _voiced in segs:
            alpha, beta, d = gen_params(n, dist_range, rng)
            new_spec[fs:fe + 1] = _warp_frames(
                frames[fs:fe + 1], omega,
                lambda w, a=alpha, b=beta: multi_warp(w, a, b),
            )
            archive.append(SegmentParams(fs, fe, alpha, beta, d))
        out_frames = np.fft.irfft(new_spec, n=FRAME_LEN, axis=-1)
        return out_frames.ravel()[: wav.size], archive

    for seg in archive:
        new_spec[seg.frame_start:seg.frame_end + 1] = _warp_frames(
            frames[seg.frame_start:seg.frame_end + 1], omega,
            lambda w, a=seg.alpha, b=seg.beta: inverse_multi_warp(w, a, b),
        )
    out_frames = np.fft.irfft(new_spec, n=FRAME_LEN, axis=-1)
    return out_frames.ravel()[: wav.size]


def protect(wav, n: int | None = None, dist_range=DIST_RANGE, rng=None):
    """声音匿名化（论文 §4.2）。返回 (protected_wav, archive)，
    archive 为逐段 SegmentParams 列表，即恢复所需的密钥材料（论文的参数集 S）。"""
    return _apply(wav, n=n, dist_range=dist_range, rng=rng, invert=False)


def recover(wav_protected, archive: list[SegmentParams]):
    """按参数档案恢复（论文 §4.4）：与 protect 同一管线，映射换成 m⁻¹。"""
    return _apply(wav_protected, archive=archive, invert=True)
