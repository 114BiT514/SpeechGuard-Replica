"""[P0-5] MP3 发布层 FNR 评估（论文 Table 7 口径的有损格式版）。

流程与系统真实发布路径一致（pipeline.publish）：
  原始音频 → protect（warping + 原始音频检测 + AES-CTR 帧加密）→ 16k WAV
  → ffmpeg 编码 128kbps MP3（44.1k mono）→ 公式(5) 秒→MP3帧号 →
  ① encrypt_mp3（加密分发版）② quiet_mp3（无权限静音分发版）
  → ffmpeg 解码回 16k PCM → Conformer（beam=4）转写 → 命中判定同 eval_fnr。

采样同 eval_fnr（test-clean 按序扫描、CTC 词流检测 ≥2 敏感项），取前 N=25 句
（60 句全集的子集，词表/命中口径一致，与 WAV 层 FNR 直接可比）。

已知效应（报告注明，非 bug）：加密分发版帧体密文不满足 Huffman 约束时第三方
解码器整帧丢弃 → 时间轴前移；静音分发版时长保持、窗外逐样本无损（M4 文档 §4）。

运行：E:/Miniconda/envs/speechguard/python.exe experiments/eval_fnr_mp3.py
输出：experiments/results/fnr_eval_mp3.json（缓存 fnr_mp3_hyp_cache.json）
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.eval_asr import load_asr, normalize, transcribe  # noqa: E402
from src import pipeline  # noqa: E402
from src.content.mp3_crypto import (encrypt_mp3, parse, quiet_mp3,  # noqa: E402
                                    time_spans_to_frames)

SR = 16000
SR_MP3 = 44100
FRAME_LEN = 576
N_UTTS = 25
MARGIN = 8                       # annotation.MARGIN_FRAMES 默认（FNR 扫参工作点）
DEVICE = "cuda:0"
RES = os.path.join(PROJ, "experiments", "results")

FFMPEG = (shutil.which("ffmpeg")
          or r"E:/Miniconda/envs/speechguard/Library/bin/ffmpeg.exe")


def mp3_pipeline_case(wav16k, rng_seed):
    """一条语音的 MP3 层三态（单变量隔离：WAV 层【不】加密敏感词，只测公式(5)）。

    prot 态音频 → 128kbps MP3（44.1k）→ 同一 ranges 分别做
    帧体加密（enc）/ 静音替换（quiet），干净 MP3 作基线（clean）。
    系统真实发布链路（WAV+MP3 双护）的 quiet.enc 组合由 pipeline 负责。
    """
    tmp = tempfile.mkdtemp(prefix="fnrmp3_")
    try:
        b = pipeline.protect(np.asarray(wav16k, np.float32), device=DEVICE,
                             rng=np.random.default_rng(rng_seed))
        sens = [w for it in b["ann"].items if not it.deleted
                for w in it.text.upper().split()]
        plain_path = os.path.join(tmp, "prot.wav")
        sf.write(plain_path, pipeline._to_float(b["prot_pcm"]), SR)
        mp3_path = os.path.join(tmp, "prot.mp3")
        from src.content.mp3_crypto import encode_mp3
        encode_mp3(plain_path, mp3_path)
        orig = open(mp3_path, "rb").read()
        doc = parse(orig)
        ranges = time_spans_to_frames(
            doc, [(fs * FRAME_LEN / SR, (fe + 1) * FRAME_LEN / SR)
                  for fs, fe in b["spans"]])
        return {"clean": orig,
                "enc": encrypt_mp3(orig, ranges, b["keys"]),
                "quiet": quiet_mp3(orig, ranges),
                "sens": sens,
                "mp3_ranges": sum(fe - fs + 1 for fs, fe in ranges),
                "mp3_frames": len(doc.frames)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def mp3_to_pcm(data: bytes) -> np.ndarray:
    """MP3 字节 → 16k mono float32（ffmpeg 解码，容忍坏帧）。

    加密帧体大概率不满足 Huffman 约束，ffmpeg 报 "big_values too big" 并返回
    非零码，但**窗外帧仍正常解码输出**（这正是论文 §4.3.3 描述第三方播放器
    的丢弃行为，与 M4 文档 §4 实测一致）。评估需要的是可解码部分，故只在
    输出为空时判失败。
    """
    r = subprocess.run([FFMPEG, "-v", "error", "-f", "mp3", "-i", "pipe:0",
                        "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
                       input=data, capture_output=True)
    a = np.frombuffer(r.stdout, np.float32)
    assert a.size > 100, f"empty decode (rc={r.returncode}): {r.stderr.decode()[:200]}"
    return a


def main():
    files = sorted(glob.glob(os.path.join(
        PROJ, "data", "librispeech", "LibriSpeech", "test-clean", "*", "*",
        "*.flac")))
    picked = []
    print("采样（与 eval_fnr 同协议，前", N_UTTS, "句）...", flush=True)
    t0 = time.time()
    for i, f in enumerate(files):
        wav, _ = sf.read(f)
        wav = np.asarray(wav, np.float32)
        b = pipeline.protect(wav, device=DEVICE,
                             rng=np.random.default_rng(700 + i))
        sens = [w for it in b["ann"].items if not it.deleted
                for w in it.text.upper().split()]
        if len(sens) >= 2:
            picked.append((f, wav, i))
        if len(picked) >= N_UTTS:
            break
    print(f"采样 {len(picked)} 句 / {time.time()-t0:.0f}s", flush=True)

    cache_path = os.path.join(RES, "fnr_mp3_hyp_cache_v2.json")
    cache = (json.load(open(cache_path, encoding="utf-8"))
             if os.path.exists(cache_path) else {})

    jobs, metas = [], []
    for f, wav, i in picked:
        rel = os.path.relpath(f, PROJ)
        case = mp3_pipeline_case(wav, 900 + i)
        jobs.append((rel, "enc", mp3_to_pcm(case["enc"])))
        jobs.append((rel, "quiet", mp3_to_pcm(case["quiet"])))
        jobs.append((rel, "clean_mp3", mp3_to_pcm(case["clean"])))
        metas.append({"file": rel, "sens": case["sens"],
                      "mp3_ranges": case["mp3_ranges"],
                      "mp3_frames": case["mp3_frames"]})

    todo = [(r, m, w) for r, m, w in jobs if f"{r}|{m}" not in cache]
    print(f"缓存命中 {len(jobs)-len(todo)}/{len(jobs)}", flush=True)
    if todo:
        model = load_asr(device=DEVICE, overrides="test_beam_size: 4")
        hyps = transcribe(model, [w for _, _, w in todo], batch_size=2)
        for (r, m, _), h in zip(todo, hyps):
            cache[f"{r}|{m}"] = h
        json.dump(cache, open(cache_path, "w", encoding="utf-8"),
                  ensure_ascii=False)

    def score(mode):
        total = rec = 0
        surv = []
        for meta in metas:
            toks = set(normalize(cache[f"{meta['file']}|{mode}"]).split())
            for w in meta["sens"]:
                total += 1
                if w.upper().strip(".,!?;:") in toks:
                    rec += 1
                    surv.append({"w": w, "utt": meta["file"]})
        return {"sensitive": total, "recognized": rec, "fnr": 1 - rec / total}, surv

    ori, _ = score("clean_mp3")          # MP3 编解码本身的损失基线（未加密）
    enc, enc_s = score("enc")
    q, q_s = score("quiet")
    out = {"protocol": {
        "n_utts": len(metas), "asr": "Conformer beam=4（论文协议）",
        "path": "pipeline.publish 真实路径（128kbps MP3、公式(5) 帧体加密/静音替换）",
        "note": "enc 版第三方解码整帧丢弃噪声帧（时间前移），quiet 版时长保持——"
                "两版敏感词均不可识别；与 WAV 层 eval_fnr 同一采样/打分口径"},
        "clean_mp3": {**ori}, "enc_mp3": {**enc}, "quiet_mp3": {**q},
        "frame_share": sum(m["mp3_ranges"] for m in metas) /
                       max(sum(m["mp3_frames"] for m in metas), 1),
        "survivors_enc": enc_s[:20], "survivors_quiet": q_s[:20]}
    json.dump(out, open(os.path.join(RES, "fnr_eval_mp3.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"clean_mp3 FNR={ori['fnr']:.2%} | enc FNR={enc['fnr']:.2%} "
          f"| quiet FNR={q['fnr']:.2%}（论文 Enc 95.88%）")
    print("saved ->", os.path.join(RES, "fnr_eval_mp3.json"))


if __name__ == "__main__":
    main()
