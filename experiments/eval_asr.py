"""[M3 任务3] ASR（Conformer）评估工具：转写、文本归一、WER。

论文 §5.1：Conformer ASR（LibriSpeech 960h 训练）转写，WER 衡量语音质量/可懂度
（越低越好）。注意 SpeechBrain 路径须用正斜杠（Windows 反斜杠会被吃掉，见 smoke_test）。
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)


def load_asr(device: str = "cuda:0", savedir: str | None = None,
             overrides: str | None = None):
    from speechbrain.inference.ASR import EncoderDecoderASR
    from speechbrain.utils.fetching import LocalStrategy

    savedir = savedir or os.path.join(PROJ, "data", "pretrained", "conformer-asr")
    return EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-conformer-transformerlm-librispeech",
        savedir=savedir,
        run_opts={"device": device},
        overrides=overrides,
        local_strategy=(
            LocalStrategy.COPY_SKIP_CACHE if sys.platform == "win32" else LocalStrategy.SYMLINK
        ),
    )


def normalize(text: str) -> str:
    """LibriSpeech 规范：大写，仅保留 A-Z、撇号与空格。"""
    t = re.sub(r"[^A-Z' ]", " ", text.upper())
    return re.sub(r"\s+", " ", t).strip()


def librispeech_ref(flac_path: str) -> str:
    """LibriSpeech 布局：每个书目录一个 `<spk>-<book>.trans.txt`，
    每行 `<utterance-id> TEXT...` → 按语音 ID 查行并归一化。"""
    import glob as _glob

    folder = os.path.dirname(flac_path)
    trans_files = _glob.glob(os.path.join(folder, "*.trans.txt"))
    if not trans_files:
        raise FileNotFoundError(f"no .trans.txt in {folder}")
    utt_id = os.path.splitext(os.path.basename(flac_path))[0]
    with open(trans_files[0], encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if parts[0] == utt_id:
                return normalize(parts[1])
    raise KeyError(f"{utt_id} not found in {trans_files[0]}")


def transcribe(model, wavs: list[np.ndarray], batch_size: int = 2) -> list[str]:
    """16kHz mono 波形列表 → 转写文本列表。

    按长度排序分批（减少 padding），batch 保持很小：Conformer 的 beam search+LM
    在长语音上显存开销大（8GB 卡上 batch=8×35s 会 OOM）。每批后清理显存碎片。
    """
    import torch

    order = np.argsort([len(w) for w in wavs])
    hyps: list[str] = [""] * len(wavs)
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            batch = [wavs[i] for i in idx]
            lens = torch.tensor([len(w) for w in batch], dtype=torch.float32)
            max_len = int(lens.max())
            mat = torch.zeros(len(batch), max_len)
            for j, w in enumerate(batch):
                mat[j, : len(w)] = torch.as_tensor(np.asarray(w, np.float32))
            rel = lens / max_len
            texts, _ = model.transcribe_batch(mat.to(model.device), rel.to(model.device))
            for j, i in enumerate(idx):
                hyps[i] = texts[j]
            del mat, texts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return hyps


def corpus_wer(model, wavs: list[np.ndarray], refs: list[str], batch_size: int = 2):
    """语料级 WER（总编辑数 / 总参考词数）与逐句 WER 列表。"""
    import jiwer

    hyps = transcribe(model, wavs, batch_size)
    refs_n = [normalize(r) for r in refs]
    hyps_n = [normalize(h) for h in hyps]
    per_utt = [
        jiwer.wer(r, h) if r.strip() or h.strip() else 0.0
        for r, h in zip(refs_n, hyps_n)
    ]
    total_ref = sum(len(r.split()) for r in refs_n)
    edits = sum(round(w * len(r.split())) for w, r in zip(per_utt, refs_n))
    corpus = edits / total_ref if total_ref else float("nan")
    return corpus, per_utt, hyps_n
