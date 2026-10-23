"""[M2 任务1] 36ms 帧切分与帧号-时间戳换算。

论文 §4.1：SpeechGuard 以帧为单位处理音频，每帧 36 ms（16kHz 下 576 采样点）。
帧号与时间的换算是后续所有模块的基础：VAD 分段、warping 参数按段（segment）分配、
敏感词加密的帧区间映射（论文公式(5)的同族逻辑）都建立在这里。

约定：
- 帧 i 覆盖采样点 [i*hop, i*hop + frame_len)，时间 [i*hop/sr, (i*hop+frame_len)/sr)；
- hop 默认等于 frame_len（无重叠，与论文公式(5)的 MP3 帧语义一致）；
  STFT 模块用 hop=frame_len//2 的重叠帧化，复用本模块的换算函数。
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 36
FRAME_LEN = int(SAMPLE_RATE * FRAME_MS / 1000)  # 576


def frame_count(n_samples: int, frame_len: int = FRAME_LEN, hop: int | None = None) -> int:
    """覆盖 n_samples 个采样点所需的帧数（末帧允许越界，由补零兜底）。"""
    hop = frame_len if hop is None else hop
    if n_samples <= 0:
        return 0
    if n_samples <= frame_len:
        return 1
    return 1 + int(np.ceil((n_samples - frame_len) / hop))


def make_frames(wav, frame_len: int = FRAME_LEN, hop: int | None = None):
    """一维波形 → 帧矩阵 (n_frames, frame_len)。

    末尾补零凑整帧，返回 (frames, pad)：还原时取 frames.ravel()[:n_samples] 即可
    （n_samples 为原始长度，pad 为补零点数）。
    """
    hop = frame_len if hop is None else hop
    wav = np.asarray(wav, dtype=np.float64).ravel()
    n = wav.size
    n_frames = frame_count(n, frame_len, hop)
    pad = (n_frames - 1) * hop + frame_len - n if n_frames else 0
    if pad > 0:
        wav = np.pad(wav, (0, pad))
    frames = np.lib.stride_tricks.sliding_window_view(wav, frame_len)[::hop].copy()
    return frames, int(pad)


def frame_span(frame_idx: int, hop: int | None = None, frame_len: int = FRAME_LEN,
               sr: int = SAMPLE_RATE) -> tuple[float, float]:
    """帧 i 覆盖的时间区间 (start_sec, end_sec)，左闭右开。"""
    hop = frame_len if hop is None else hop
    start = frame_idx * hop / sr
    return start, start + frame_len / sr


def frames_for_span(t_start: float, t_end: float, hop: int | None = None,
                    frame_len: int = FRAME_LEN, sr: int = SAMPLE_RATE) -> tuple[int, int]:
    """时间区间 [t_start, t_end] 覆盖到的帧号闭区间 (fs, fe)。

    取整方向与论文公式(5)一致（起点向下取整、终点向上取整），
    对加密等破坏性操作"宁多勿漏"：te 恰好落在帧边界时，边界帧也会被包含。
    """
    hop = frame_len if hop is None else hop
    fs = int(np.floor(t_start * sr / hop))
    fe = int(np.ceil(t_end * sr / hop))
    return max(fs, 0), fe
