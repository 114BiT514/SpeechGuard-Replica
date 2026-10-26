"""M5 任务5 基线单测：三种频域 warp + VoiceMask 时域方法的往返可逆性与密钥约束。"""
import numpy as np
import pytest

from src.baselines import methods as B
from src.baselines import voicemask as VM


@pytest.fixture()
def sig():
    rng = np.random.default_rng(3)
    t = np.arange(16000 * 1.2) / 16000
    return (0.3 * np.sin(2 * np.pi * 300 * t)
            + 0.05 * rng.standard_normal(t.size))


@pytest.mark.parametrize("method", ["single", "mcadams", "overo"])
def test_warp_roundtrip_bounded_error(sig, method):
    p, arch = B.protect(sig, method, rng=np.random.default_rng(5))
    r = B.recover(p, arch, method)
    n = min(len(r), len(sig))
    err = float(np.mean((r[:n] - sig[:n]) ** 2) / np.mean(sig[:n] ** 2))
    # 合成谐波+噪声是波形级往返的最坏情况（M2 同款注记），基线无斜率约束失真更大；
    # 这里只要求"真参数恢复 ≪ 完全未恢复"，特征级质量由 exp4/exp6 评估给出
    assert err < 0.6, f"{method} 往返失真 {err:.3f} 过大"
    raw = float(np.mean((p[:n] - sig[:n]) ** 2) / np.mean(sig[:n] ** 2))
    assert err < raw, "恢复未比保护态更接近原始"


def test_warp_wrong_param_fails(sig):
    p, arch = B.protect(sig, "single", rng=np.random.default_rng(6))
    r_ok = B.recover(p, arch, "single")
    r_bad = B.recover(p, [2.0], "single")        # 错 α
    n = min(len(r_ok), len(sig))
    e_ok = float(np.mean((r_ok[:n] - sig[:n]) ** 2))
    e_bad = float(np.mean((r_bad[:n] - sig[:n]) ** 2))
    assert e_bad > 3 * e_ok


def test_voicemask_exact_roundtrip(sig):
    rng = np.random.default_rng(1)
    p, params = VM.protect(sig, rng=rng)
    rec = VM.recover(p, params)
    np.testing.assert_allclose(rec, sig, atol=1e-12)   # 时域反转精确可逆
    assert VM.distortion_to_original(p, sig) > 0.5      # masked 态远离原始


def test_voicemask_segment_len_in_bounds():
    rng = np.random.default_rng(2)
    for _ in range(50):
        _, (L,) = VM.protect(np.zeros(16000), rng=rng)
        assert VM.L_RANGE[0] <= L <= VM.L_RANGE[1]
