"""M2 预处理单元测试：framing 分帧/时间换算、stft 完美重构。

运行（仓库根目录）：E:/Miniconda/envs/speechguard/python.exe -m pytest tests -q
"""
import numpy as np

from src.preprocess.framing import (
    FRAME_LEN,
    FRAME_MS,
    SAMPLE_RATE,
    frame_count,
    frame_span,
    frames_for_span,
    make_frames,
)
from src.preprocess.stft import CENTER_PAD, DEFAULT_HOP, istft, sqrt_hann, stft

rng = np.random.default_rng(0)


def test_constants():
    assert FRAME_MS == 36
    assert SAMPLE_RATE == 16000
    assert FRAME_LEN == 576            # 16000 * 0.036
    assert DEFAULT_HOP == 288     # 50% 重叠


def test_frame_count():
    assert frame_count(0) == 0
    assert frame_count(1) == 1
    assert frame_count(FRAME_LEN) == 1
    assert frame_count(FRAME_LEN + 1) == 2
    assert frame_count(2 * FRAME_LEN) == 2
    assert frame_count(2 * FRAME_LEN + 1) == 3
    # 重叠帧化（STFT 用）
    assert frame_count(FRAME_LEN, hop=DEFAULT_HOP) == 1
    assert frame_count(16000 + CENTER_PAD, hop=DEFAULT_HOP) == 1 + int(
        np.ceil((16000 + CENTER_PAD - FRAME_LEN) / DEFAULT_HOP)
    )


def test_make_frames_roundtrip_no_overlap():
    x = rng.standard_normal(1000)
    frames, pad = make_frames(x)
    assert frames.shape == (2, FRAME_LEN)
    assert pad == 2 * FRAME_LEN - 1000
    np.testing.assert_allclose(frames.ravel()[:1000], x)


def test_make_frames_exact_multiple():
    x = rng.standard_normal(2 * FRAME_LEN)
    frames, pad = make_frames(x)
    assert frames.shape == (2, FRAME_LEN)
    assert pad == 0
    np.testing.assert_allclose(frames.ravel(), x)


def test_make_frames_overlap():
    x = rng.standard_normal(1600)
    frames, pad = make_frames(x, hop=DEFAULT_HOP)
    assert frames.shape == (frame_count(1600, hop=DEFAULT_HOP), FRAME_LEN)
    assert frames.size >= 1600


def test_frame_span():
    assert frame_span(0) == (0.0, 0.036)
    np.testing.assert_allclose(frame_span(2), (0.072, 0.108), atol=1e-12)


def test_frames_for_span():
    # 取整方向与论文公式(5)一致：宁多勿漏
    assert frames_for_span(0.0, 0.036) == (0, 1)      # te 恰在帧边界，边界帧包含
    assert frames_for_span(0.036, 0.072) == (1, 2)
    assert frames_for_span(0.01, 0.02) == (0, 1)
    assert frames_for_span(0.0, 0.0359) == (0, 1)
    assert frames_for_span(-0.01, 0.02) == (0, 1)     # 负时间夹紧到 0


def test_sqrt_hann():
    w = sqrt_hann(FRAME_LEN)
    assert w.shape == (FRAME_LEN,)
    assert w[0] == 0.0
    np.testing.assert_allclose(w[FRAME_LEN // 2], 1.0)
    # 周期性：首尾同为 0，50% 重叠处两窗平方和为 1（COLA）
    np.testing.assert_allclose(w[: DEFAULT_HOP] ** 2 + w[DEFAULT_HOP:] ** 2, 1.0, atol=1e-12)


def test_stft_shape():
    x = rng.standard_normal(16000)
    spec, pad = stft(x)
    n_expected = frame_count(16000 + CENTER_PAD, hop=DEFAULT_HOP)
    assert spec.shape == (n_expected, FRAME_LEN // 2 + 1)
    assert np.iscomplexobj(spec)


def test_stft_istft_perfect_reconstruction():
    """可逆性的地基：istft(stft(x)) ≈ x（误差应为浮点级）。"""
    for n in (576, 1600, 16000):
        x = rng.standard_normal(n)
        spec, pad = stft(x)
        y = istft(spec, target_len=n)
        err = float(np.max(np.abs(y - x)))
        assert err < 1e-9, f"n={n}, max_err={err}"


def test_stft_istft_nonoverlap_rectangular():
    """hop=frame_len + 全 1 窗：退化为无重叠精确重构（对应公式(5)的帧语义）。"""
    x = rng.standard_normal(1600)
    rect = np.ones(FRAME_LEN)
    spec, pad = stft(x, hop=FRAME_LEN, window=rect)
    y = istft(spec, hop=FRAME_LEN, window=rect, target_len=1600)
    np.testing.assert_allclose(y, x, atol=1e-12)
