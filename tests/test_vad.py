"""M2 VAD 单元测试：公式(1) 递归阈值与分段。"""
import numpy as np

from src.preprocess.vad import segmentation, voiced_flags


def test_silence_is_unvoiced():
    """全零谱：峰值≈0，阈值收敛到 ~1.86，应判无声。"""
    spec = np.zeros((20, 289), dtype=np.complex128)
    flags = voiced_flags(spec)
    assert not flags.any()


def test_loud_tone_is_voiced():
    """少数帧有强分量（|X|=300）：应判有声，且强帧序列保持有声。"""
    spec = np.zeros((12, 289), dtype=np.complex128)
    spec[2:5, 100] = 300.0
    flags = voiced_flags(spec)
    assert list(flags) == [False, False, True, True, True,
                           False, False, False, False, False, False, False]


def test_formula_matches_paper_reference():
    """手算前几帧递归值，核对公式(1)实现（T=1, at=0.7, a1=0.25, δ=1.99）。"""
    spec = np.zeros((4, 4), dtype=np.complex128)
    spec[0, 0] = 10.0
    peaks = np.abs(spec).max(axis=1)
    # 手算：Φ(-1)=10+1.99=11.99
    # Φ(0)=0.7*(10+1.99)+0.25*11.99 = 8.393+2.9975 = 11.3905
    # Φ(1)=0.7*1.99+0.25*11.3905 = 1.393+2.847625 = 4.240625
    flags = voiced_flags(spec)
    assert flags[0] == (10.0 >= 11.3905)      # False
    assert flags[1] == (0.0 >= 4.240625)      # False
    assert flags[2] is False or flags[2] == False  # 持续无声


def test_segmentation_groups_runs():
    flags = np.array([False, True, True, True, False, False, True, False])
    segs = segmentation(flags)
    assert segs == [(0, 0, False), (1, 3, True), (4, 5, False), (6, 6, True), (7, 7, False)]


def test_segmentation_edges():
    assert segmentation(np.array([True])) == [(0, 0, True)]
    assert segmentation(np.array([False])) == [(0, 0, False)]
    assert segmentation(np.array([True] * 9)) == [(0, 8, True)]
    # 覆盖完备性：段恰好铺满且互不重叠
    rng = np.random.default_rng(1)
    flags = rng.integers(0, 2, 100).astype(bool)
    segs = segmentation(flags)
    assert segs[0][0] == 0 and segs[-1][1] == 99
    for (s1, e1, _), (s2, e2, _) in zip(segs, segs[1:]):
        assert e1 + 1 == s2
