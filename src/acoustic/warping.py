"""[M2 任务4] 多参数分段线性频域扭曲（warping）——全项目技术核心。

论文 §4.2：
- 公式(2)：单参数两段线性扭曲（对比基线）。ω0 = 7π/8（α≤1）或 7π/(8α)（α>1）；
- 公式(3)：多参数扭曲 m(ω,α,β)：穿过点列 (0,0),(α1,β1),…,(αn,βn),(π,π) 的分段线性
  函数，α、β 严格递增且在 (0,π) 内（边界 α0=β0=0, α_{n+1}=β_{n+1}=π）→ m 严格
  递增 → [0,π]→[0,π] 的双射；
- 逆函数：m⁻¹(ω,α,β) = m(ω,β,α)（α、β 互换），这是"可恢复"的全部来源；
- 公式(4)：扭曲强度 dist = ∫₀^π |F(ω)−ω| dω；论文 §5.1 限定 dist∈(0.5,0.6)、n∈[1,10]。

实现说明与已知局限（完整证据链见 scripts/variant_probe*.py 与 M3 实验，报告将展开）：
- **块级处理（默认）**：无重叠 36ms 块（hop=FRAME_LEN，与论文公式(5)帧语义一致），
  块 → DFT → 逐段扭曲（段内各帧共用参数）→ IDFT。
- **精细网格必须整除覆盖 DFT 网格**：FINE_N = 16×FRAME_LEN。若不成整数倍
  （如 8192/576=14.22），恒等映射也要跨网格缝隙插值，roundtrip 直接崩坏
  （实测恒等仅 13dB；修复后恒等 ≈300dB）。
- **斜率约束**：论文只约束 α、β 严格递增与 dist 范围；纯随机抽样会以可观概率产生
  数十倍局部斜率。此处约束分段斜率 ∈ [0.5, 2]（VTLN 常规范围）。代价：抗
  reducing 攻击成功率从论文的 ≈16-18% 升到 ≈42%（M3 exp2 已量化）。
- **参数构造**：Dirichlet 切段 + 斜率偏差加权中心化（保证终点约束与斜率界限严格
  成立），仅对 dist 做拒绝采样；独立排序抽样在斜率受限时对 n≥8 几乎不可行。
- **实现变体的 EER/WER 权衡（M3 实测，dev-clean 8人×3条，ECAPA/Conformer）**：
  块级 rect（本默认）：WER 30.6% / EER 37.5% —— EER 与论文 40.09% 对齐，
  WER 远差于论文 3.58%（块边界伪影）；
  段级整体 DFT（仅有声）：WER 13.6% / EER 1.4%；
  OLA 加窗合成：WER 2.6% / EER 3.3%（合成干净但 ECAPA 对平稳扭曲近乎免疫）；
  细粒度不合并：WER 93% / EER 22.5%（破坏式）。
  **结论：干净频域扭曲无法同时复现论文的 (WER 3.58%, EER 40%) 组合——ECAPA 对
  温和频域扭曲近乎免疫，论文的高 EER 很可能依赖其实现的处理损伤。** 离散化
  不可逆与该权衡构成本复现的核心发现（对应论文自述"interpolation result in
  differences"，恢复 EER 7.07%/WER 9.90% 亦有损）。
  后续检验（scripts/vocoder_resynth + variant_probe）：
  - 假说1"扭曲+声码器重建"：证伪。块级扭曲→HiFi-GAN 重合成 WER=3.27%（优于
    论文）但 EER=0.83%——声码器洗掉块伪影后，暴露出频域扭曲本身对 ECAPA 无效。
  - 假说3"斜率约束过温和"：证伪。论文原味抽样（斜率达 11.3）块级 EER=25%，
    反而低于约束版 37.5%。
  - 假说2"TitaNet-Large 对扭曲更敏感"：证伪。TitaNet 对块级保护音频 EER=0.0%
    （ECAPA 同音频 37.5%）——比 ECAPA 更鲁棒；genuine 相似度 0.855 vs 原始 0.864。
  **最终结论：论文 (WER 3.58%, EER 40%) 工作点无法由披露公式的任何忠实实现复现，
  详见 docs/M3_reproduction_findings.md（含完整证据表与作者邮件行动项）。**
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


def distortion_strength(alpha=None, beta=None, warp=None, n_grid: int = 20001,
                        grid: np.ndarray | None = None) -> float:
    """公式(4)：dist = ∫₀^π |F(ω)−ω| dω（梯形数值积分）。identity 时为 0。

    warp(grid) 应返回与 grid 同形状的映射值；传入 grid 时 n_grid 被忽略。
    """
    grid = np.linspace(0.0, PI, n_grid) if grid is None else np.asarray(grid, dtype=np.float64)
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


def _apply(wav, archive=None, n=None, dist_range=DIST_RANGE, rng=None, invert=False,
           param_pool=None):
    """protect/recover 共用管线：invert=False 扭曲（保护），True 逆扭曲（恢复）。

    param_pool: [(alpha, beta, dist)] 预生成参数池——dist 高端区间的拒绝采样较慢，
    批量实验时按 (n, dist_range) 预构建复用；每个段从池中随机抽取。
    """
    wav = np.asarray(wav, dtype=np.float64).ravel()
    rng = np.random.default_rng() if rng is None else rng
    frames, spec, omega = _frame_view(wav)
    new_spec = spec.copy()

    if not invert:
        flags = voiced_flags(spec)
        segs = merge_short_segments(segmentation(flags), MIN_SEG_FRAMES)
        archive = []
        for fs, fe, _voiced in segs:
            if param_pool:
                alpha, beta, d = param_pool[rng.integers(len(param_pool))]
            else:
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


def _apply_ola(wav, archive=None, n=None, dist_range=DIST_RANGE, rng=None, invert=False,
               param_pool=None):
    """OLA 模式：加窗重叠帧（sqrt-Hann, hop=288）→ 逐段扭曲帧谱 → WSS 归一化合成。

    合成平滑（无块边界伪影，WER≈原始），扭曲仍逐段独立。archive 帧号语义与
    block 模式一致（576-hop 网格），内部 ×2 映射到 288-hop 帧。
    """
    from src.preprocess.stft import DEFAULT_HOP, frames_centered, istft, sqrt_hann

    wav = np.asarray(wav, dtype=np.float64).ravel()
    rng = np.random.default_rng() if rng is None else rng
    frames, _pad = frames_centered(wav)
    w = sqrt_hann(FRAME_LEN)
    spec = np.fft.rfft(frames * w, axis=-1)
    omega = np.linspace(0.0, PI, spec.shape[1])
    omega_fine = np.linspace(0.0, PI, FINE_N // 2 + 1)
    new_spec = spec.copy()

    def warp_frames(i0, i1, alpha, beta, inverse):
        fmap = inverse_multi_warp if inverse else multi_warp
        for i in range(i0, i1):
            fine = np.fft.rfft(frames[i] * w, n=FINE_N)
            tgt = fmap(omega, alpha, beta)
            new_spec[i] = (np.interp(tgt, omega_fine, fine.real)
                           + 1j * np.interp(tgt, omega_fine, fine.imag))

    if not invert:
        flags = voiced_flags(np.fft.rfft(make_frames(wav)[0], axis=-1))
        segs = merge_short_segments(segmentation(flags), MIN_SEG_FRAMES)
        archive = []
        for fs, fe, _voiced in segs:
            if param_pool:
                alpha, beta, d = param_pool[rng.integers(len(param_pool))]
            else:
                alpha, beta, d = gen_params(n, dist_range, rng)
            warp_frames(fs * 2, min((fe + 1) * 2, spec.shape[0]), alpha, beta, False)
            archive.append(SegmentParams(fs, fe, alpha, beta, d))
        return istft(new_spec, target_len=wav.size), archive

    for seg in archive:
        warp_frames(seg.frame_start * 2, min((seg.frame_end + 1) * 2, spec.shape[0]),
                    seg.alpha, seg.beta, True)
    return istft(new_spec, target_len=wav.size)


def _apply_single(wav, archive=None, n=None, dist_range=DIST_RANGE, rng=None,
                  invert=False, param_pool=None):
    """single 模式：整句一个平稳映射 + OLA 加窗合成（向量化插值矩阵实现）。

    动机（M3 规模实验后）：每段独立映射会在句内切换频率映射，共振峰轨迹被
    不一致地搬移 → ASR 混淆累积（全集 WER 24-43%）。整句单一映射：句内完全
    平滑（WER 接近原始），句间映射不同 → 大样本下说话人内变异照样撑高 EER。
    """
    from src.preprocess.stft import frames_centered, istft, sqrt_hann

    wav = np.asarray(wav, dtype=np.float64).ravel()
    rng = np.random.default_rng() if rng is None else rng
    frames, _pad = frames_centered(wav)
    win = frames * sqrt_hann(FRAME_LEN)
    fine = np.fft.rfft(win, n=FINE_N, axis=-1)          # (F, FINE_N//2+1)
    omega_fine = np.linspace(0.0, PI, FINE_N // 2 + 1)
    omega = np.linspace(0.0, PI, FRAME_LEN // 2 + 1)

    if not invert:
        alpha, beta, d = (param_pool[rng.integers(len(param_pool))] if param_pool
                          else gen_params(n, dist_range, rng))
        archive = [SegmentParams(0, frames.shape[0] - 1, alpha, beta, d)]
        tgt = multi_warp(omega, alpha, beta)
    else:
        seg = archive[0]
        tgt = inverse_multi_warp(omega, seg.alpha, seg.beta)

    # 线性插值矩阵 S: Y = fine @ Sᵀ（单映射 → 对全帧共享，向量化）
    idx = np.clip(np.searchsorted(omega_fine, tgt, side="right") - 1,
                  0, len(omega_fine) - 2)
    denom = omega_fine[idx + 1] - omega_fine[idx]
    w1 = (tgt - omega_fine[idx]) / denom
    S = np.zeros((len(tgt), len(omega_fine)))
    S[np.arange(len(tgt)), idx] = 1.0 - w1
    S[np.arange(len(tgt)), idx + 1] = w1
    new_spec = fine @ S.T                                # (F, 289)
    return istft(new_spec, target_len=wav.size), archive


def protect(wav, n: int | None = None, dist_range=DIST_RANGE, rng=None,
            param_pool: list | None = None, mode: str = "single"):
    """声音匿名化（论文 §4.2）。返回 (protected_wav, archive)，
    archive 为 SegmentParams 列表，即恢复所需的密钥材料（论文的参数集 S）。

    param_pool: 预生成的 [(alpha, beta, dist)] 池；给出时每段从池中抽取，
    忽略 n 与 dist_range（池生成时已固定）。
    mode（M3 全集实验确定的三种实现，默认 single——与论文工作点对齐）：
    - "single"：整句单一平稳映射 + OLA 加窗合成。WER 5.44% / EER 40.01%
      （论文 4.80% / 34.50%）——句内平滑保可懂度，句间随机保匿名性。
    - "ola"：逐段映射 + OLA 合成。WER 23.77% / EER 29.93%。
    - "block"：逐段映射 + 无重叠矩形块。WER 42.98% / EER 32.98%。
    """
    fn = {"block": _apply, "ola": _apply_ola, "single": _apply_single}[mode]
    return fn(wav, n=n, dist_range=dist_range, rng=rng, invert=False,
              param_pool=param_pool)


def recover(wav_protected, archive: list[SegmentParams], mode: str = "single"):
    """按参数档案恢复（论文 §4.4）：与 protect 同一管线，映射换成 m⁻¹。"""
    fn = {"block": _apply, "ola": _apply_ola, "single": _apply_single}[mode]
    if mode == "single":
        out, _ = fn(wav_protected, archive=archive, invert=True)
        return out
    return fn(wav_protected, archive=archive, invert=True)
