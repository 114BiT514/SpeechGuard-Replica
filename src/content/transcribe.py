"""[M4 任务1] wav2vec2 离线转写（论文 §4.3.1 的 SD 组件）。

论文用 WavLM 做转写；Hub 上 facebook/wavlm-large 已被 gate，改用
facebook/wav2vec2-large-960h-lv60-self（论文 Table 3 的对比模型之一，
clean 错误率 5.00% vs WavLM 4.04%）。报告中注明替换。
"""
from __future__ import annotations

import os

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_ID = "facebook/wav2vec2-large-960h-lv60-self"

_cache = {}


def load_transcriber(device: str = "cuda:0"):
    """(processor, model, torch) 单例。"""
    key = device
    if key in _cache:
        return _cache[key]
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    os.environ.setdefault("HF_HOME", os.path.join(PROJ, "data", "hf-cache"))
    proc = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device).eval()
    _cache[key] = (proc, model, torch)
    return _cache[key]


def transcribe(wavs: list[np.ndarray], device: str = "cuda:0",
               batch_size: int = 4) -> list[str]:
    """16kHz mono 波形列表 → 大写文本列表（词以空格分隔）。"""
    proc, model, torch = load_transcriber(device)
    out: list[str] = []
    with torch.no_grad():
        for i in range(0, len(wavs), batch_size):
            batch = [np.asarray(w, np.float32) for w in wavs[i : i + batch_size]]
            inputs = proc(batch, sampling_rate=16000, return_tensors="pt",
                          padding=True).to(device)
            logits = model(**inputs).logits
            ids = logits.argmax(-1)
            texts = proc.batch_decode(ids, skip_special_tokens=True)
            out.extend(t.upper() for t in texts)
    return out


def emissions(wav: np.ndarray, device: str = "cuda:0"):
    """单条音频 → (log_probs (T,V), processor) 供 forced_align 使用。"""
    proc, model, torch = load_transcriber(device)
    with torch.no_grad():
        inputs = proc([np.asarray(wav, np.float32)], sampling_rate=16000,
                      return_tensors="pt").to(device)
        logits = model(**inputs).logits
    return logits.log_softmax(-1)[0].cpu(), proc


FRAME_SEC = 320 / 16000.0  # wav2vec2-large 卷积步长：20ms/帧


def transcribe_with_times(wav: np.ndarray, device: str = "cuda:0"):
    """CTC 贪心解码 + `|` 词边界 → [(word, start_sec, end_sec)]。

    词时间戳与转写同源（不依赖外部对齐器）：折叠重复 token、去 blank 后，
    以 `|` 切词；词的帧区间 = 其字符 token 占据的首/末帧。
    """
    proc, model, torch = load_transcriber(device)
    with torch.no_grad():
        inputs = proc([np.asarray(wav, np.float32)], sampling_rate=16000,
                      return_tensors="pt").to(device)
        ids = model(**inputs).logits[0].argmax(-1).cpu().numpy()
    vocab = proc.tokenizer.get_vocab()
    blank = vocab["<pad>"]
    pipe = vocab.get("|")
    inv = {v: k for k, v in vocab.items()}

    seq = []                                   # (token, frame)
    prev = None
    for t, i in enumerate(ids):
        i = int(i)
        if i != prev and i != blank:
            seq.append((i, t))
        prev = i

    out, chars, start = [], [], None
    for tid, t in seq:
        if tid == pipe:
            if chars:
                out.append(("".join(c for c, _ in chars).strip(),
                            start * FRAME_SEC, (t + 1) * FRAME_SEC))
            chars, start = [], None
        else:
            if not chars:
                start = t
            chars.append((inv.get(tid, ""), t))
    if chars:
        out.append(("".join(c for c, _ in chars).strip(),
                    start * FRAME_SEC, (seq[-1][1] + 1) * FRAME_SEC))
    return [(w, s, e) for w, s, e in out if w]
