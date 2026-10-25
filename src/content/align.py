"""[M4 任务2] 词级时间戳（论文 §4.3.1 用 MFA；本实现用 wav2vec2 CTC 的 `|`
词边界 token 直接给出词时间——纯 pip、Windows 原生、与转写同源零失配，
免 WSL/MFA。报告中注明对齐工具差异）。

align_words(wav, transcript)：把参考/检测文本与 CTC 词时间用 difflib 对齐；
CTC 漏/错的词按相邻词时间线性插值（±50ms 级误差，加密帧区间"宁多勿漏"可接受）。
"""
from __future__ import annotations

import difflib

import numpy as np


def align_words(wav16k: np.ndarray, transcript: str, device: str = "cuda:0"):
    """返回 [(word, start_sec, end_sec)]，顺序与 transcript 词序一致。"""
    from src.content.transcribe import transcribe_with_times

    ctc = transcribe_with_times(wav16k, device)          # [(word, t0, t1)]
    ref_words = transcript.upper().split()
    if not ref_words:
        return []
    if not ctc:
        n = float(wav16k.size / 16000.0)
        step = n / len(ref_words)
        return [(w, i * step, (i + 1) * step) for i, w in enumerate(ref_words)]

    ctc_words = [w.upper() for w, _, _ in ctc]
    sm = difflib.SequenceMatcher(a=ref_words, b=ctc_words, autojunk=False)
    matched = {}                                          # ref_idx -> (t0, t1)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            matched[a + k] = (ctc[b + k][1], ctc[b + k][2])

    out = []
    for i, w in enumerate(ref_words):
        if i in matched:
            out.append((w, *matched[i]))
            continue
        # 插值：取前后最近已匹配词的时间
        prev = max((j for j in matched if j < i), default=None)
        nxt = min((j for j in matched if j > i), default=None)
        if prev is not None and nxt is not None:
            t0 = matched[prev][1]
            t1 = matched[nxt][0]
            span = (t1 - t0) / (nxt - prev)
            base = t0 + (i - prev - 1) * span
            out.append((w, base, base + span))
        elif prev is not None:
            t0 = matched[prev][1] + 0.05 * (i - prev)
            out.append((w, t0, t0 + 0.25))
        elif nxt is not None:
            t1 = max(matched[nxt][0] - 0.05 * (nxt - i), 0.0)
            out.append((w, max(t1 - 0.25, 0.0), t1))
        else:
            out.append((w, 0.0, 0.0))
    return out
