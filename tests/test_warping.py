"""M2 warping 核心单元测试：公式(2)(3)(4)、逆函数、参数生成、protect/recover 闭环。"""
import numpy as np
import pytest

from src.acoustic.warping import (
    PI,
    SegmentParams,
    distortion_strength,
    gen_params,
    inverse_multi_warp,
    multi_warp,
    piecewise_warp,
    protect,
    recover,
)

rng = np.random.default_rng(0)


# ---------------- 公式(2)：单参数基线 ----------------

def test_piecewise_warp_endpoints_and_identity():
    for alpha in (0.5, 1.0, 2.0):
        grid = np.linspace(0, PI, 1001)
        w = piecewise_warp(grid, alpha)
        np.testing.assert_allclose(w[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(w[-1], PI, atol=1e-12)
    # α=1 时 ω0=7π/8，两段斜率均为 1 → 恒等映射
    grid = np.linspace(0, PI, 1001)
    np.testing.assert_allclose(piecewise_warp(grid, 1.0), grid, atol=1e-12)


def test_piecewise_warp_turning_point():
    """α=2 → ω0=7π/(8·2)；转折点处两段表达式应连续。"""
    alpha = 2.0
    w0 = 7 * PI / (8 * alpha)
    left = piecewise_warp(np.array([w0 - 1e-9]), alpha)[0]
    right = piecewise_warp(np.array([w0 + 1e-9]), alpha)[0]
    np.testing.assert_allclose(left, right, atol=1e-6)
    np.testing.assert_allclose(piecewise_warp(np.array([w0]), alpha)[0], alpha * w0, atol=1e-9)


# ---------------- 公式(3)：多参数扭曲与逆函数 ----------------

def test_multi_warp_identity():
    """α=β → 分段点落在对角线 → 恒等映射。"""
    alpha = np.array([0.5, 1.2, 2.0])
    grid = np.linspace(0, PI, 2001)
    np.testing.assert_allclose(multi_warp(grid, alpha, alpha), grid, atol=1e-12)


def test_multi_warp_passes_knots_and_monotone():
    alpha = np.array([0.4, 1.1, 2.2])
    beta = np.array([0.9, 1.6, 2.5])
    np.testing.assert_allclose(multi_warp(alpha, alpha, beta), beta, atol=1e-12)
    grid = np.linspace(0, PI, 5001)
    w = multi_warp(grid, alpha, beta)
    assert np.all(np.diff(w) > 0), "m 必须严格递增（可逆的前提）"
    np.testing.assert_allclose(w[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(w[-1], PI, atol=1e-12)


def test_inverse_is_exact():
    """m⁻¹(m(ω)) = ω：交换 α/β 是分段线性映射的精确逆。"""
    alpha = np.array([0.7, 1.5, 2.6])
    beta = np.array([1.1, 1.9, 2.8])
    grid = np.linspace(0, PI, 4001)
    roundtrip = inverse_multi_warp(multi_warp(grid, alpha, beta), alpha, beta)
    np.testing.assert_allclose(roundtrip, grid, atol=1e-9)


# ---------------- 公式(4)：dist ----------------

def test_dist_identity_zero():
    alpha = np.array([0.5, 1.5])
    assert distortion_strength(alpha, alpha) == 0.0


def test_dist_analytic_case():
    """n=1, α=π/4, β=π/2 的解析值：dist = π²/8（两段均为正偏差，可手算）。"""
    d = distortion_strength(np.array([PI / 4]), np.array([PI / 2]))
    np.testing.assert_allclose(d, PI ** 2 / 8, atol=1e-6)


# ---------------- gen_params ----------------

def test_gen_params_rejection_sampling():
    r = np.random.default_rng(42)
    for _ in range(30):
        alpha, beta, d = gen_params(rng=r)
        n = len(alpha)
        assert 1 <= n <= 10
        assert np.all(np.diff(alpha) > 0) and np.all(np.diff(beta) > 0)
        assert alpha[0] > 0 and alpha[-1] < PI and beta[0] > 0 and beta[-1] < PI
        assert 0.5 <= d <= 0.6


def test_gen_params_fixed_n():
    r = np.random.default_rng(7)
    alpha, beta, d = gen_params(n=4, rng=r)
    assert len(alpha) == len(beta) == 4
    assert 0.5 <= d <= 0.6


# ---------------- protect / recover 闭环 ----------------

def _make_signal(seconds=1.0, sr=16000):
    """谐和信号 + 轻噪声：模拟元音谐波结构，便于考察插值误差。"""
    t = np.arange(int(seconds * sr)) / sr
    x = sum(0.15 * np.sin(2 * np.pi * f * t) for f in (180, 360, 540, 900, 1500, 2800))
    x += 0.01 * rng.standard_normal(t.size)
    # 首尾各 0.2s 静音，制造有声/无声分段
    env = np.ones_like(x)
    env[: int(0.2 * sr)] = 0.0
    env[-int(0.2 * sr):] = 0.0
    return x * env


def test_protect_recover_roundtrip():
    """闭环恢复质量：与论文行为对齐（论文自述恢复有损：WER 9.9% vs 原始 2.46%），
    故断言的是"恢复显著优于错误密钥、且能量/结构基本保留"而非波形级精确。"""
    x = _make_signal()
    snrs = []
    for seed in range(5):
        prot, archive = protect(x, rng=np.random.default_rng(seed))
        rec = recover(prot, archive)
        snrs.append(10 * np.log10(np.sum(x ** 2) / max(np.sum((x - rec) ** 2), 1e-30)))
        assert len(archive) >= 3, "首尾静音+中段语音应至少分成 3 段"
        for seg in archive:
            assert isinstance(seg, SegmentParams) and 1 <= seg.n <= 10
            assert 0.5 <= seg.dist <= 0.6
    # 纯合成谐和信号是谱峰最尖锐的最坏情况；真语 medians 更高（实测 ~6dB），M3 用 EER/WER 验收
    assert np.median(snrs) >= 2.0, f"恢复 SNR 中位数过低：{np.median(snrs):.1f} dB"
    assert min(snrs) >= 0.0, f"存在灾难性恢复：{min(snrs):.1f} dB"


def test_protect_changes_spectrum():
    """保护应显著改变频谱（匿名化生效），但能量大致守恒。"""
    from src.preprocess.stft import stft
    x = _make_signal(0.6)
    prot, _ = protect(x, rng=np.random.default_rng(5))
    x = x[: prot.size]
    xo, _ = stft(x)
    xp, _ = stft(prot)
    rel_change = np.linalg.norm(np.abs(xp) - np.abs(xo)) / np.linalg.norm(np.abs(xo))
    assert rel_change > 0.2, f"保护前后频谱变化过小：{rel_change:.3f}"
    energy_ratio = np.sum(prot ** 2) / np.sum(x ** 2)
    assert 0.2 < energy_ratio < 5.0


def test_recover_with_wrong_params_fails_to_recover():
    """错误密钥（交换 α/β 当作正确参数用）无法恢复——安全性的基本检查。"""
    x = _make_signal(0.6)
    prot, archive = protect(x, rng=np.random.default_rng(9))
    bad_archive = [SegmentParams(s.frame_start, s.frame_end, s.beta, s.alpha, s.dist)
                   for s in archive]
    rec_bad = recover(prot, bad_archive)
    rec_good = recover(prot, archive)
    x = x[: prot.size]
    err_good = np.sum((x - rec_good) ** 2)
    err_bad = np.sum((x - rec_bad) ** 2)
    assert err_bad > 3 * err_good


def test_protect_fixed_n():
    x = _make_signal(0.5)
    prot, archive = protect(x, n=3, rng=np.random.default_rng(11))
    assert all(seg.n == 3 for seg in archive)
    rec = recover(prot, archive)
    snr = 10 * np.log10(np.sum(x ** 2) / max(np.sum((x - rec) ** 2), 1e-30))
    assert snr > 0.0, f"固定 n=3 的恢复 SNR 过低：{snr:.1f} dB"
