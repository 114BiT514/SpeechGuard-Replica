"""[M7 加分项 ④] 改进方案验证：加密对象完整性标签（MAC）阻断"部分明文渗漏"。

针对 exp9-F7：AES-CTR 无完整性校验，高 SNR 噪声只扰动少量字节 → 解密后大部分
明文原样还原、敏感词以瑕疵明文形态泄漏（SNR=20dB 时 FNR 从 clean 的 90% 隐藏
跌到 88.3% 被认出）。

改进（最小侵入，密钥包扩展不改算法主体）：
  publish 端为每个加密对象 j 存 tag_j = SHA256(明文帧体)[:4] 到密钥包（与 w_j 同发）；
  解密端解出明文后校验 tag，不匹配 → 该区间回退为静音（拒绝输出可能渗漏的明文）。
  语义：把"字节扰动→部分渗漏"变成"字节扰动→干净失败（静音）"，
  隐私面 FNR 恢复到 ~100%，可用性损失与无改进时相同（本来也解不对）。

对照设计（同 protect/加密种子配对）：
  channel ∈ {clean, noise_20db, noise_10db} × {naive 解密, MAC 解密}
  指标：FNR（敏感词隐藏率）、窗外内容可懂度（WER 仅 clean 应≈Ori）

附 E：quiet 分发版窗内静音度复测（排除窗边界 ±100ms 的双重转码对齐 slop，
只量核心区）——把 exp10-D 的 -24.8dB 离群值归因为边界错位并给出修正数字。

运行（GPU ~20-25 分钟，A2 采样确定性重放 + ASR 走共享缓存）：
  E:/Miniconda/envs/speechguard/python.exe experiments/exp11_mac_integrity.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.eval_asr import normalize  # noqa: E402
from experiments.exp8_channel_cases import (  # noqa: E402
    HYP_CACHE, SEED, SR, _add_noise, apply_channel, pick_a2_files,
    transcribe_cached)

RES = os.path.join(PROJ, "experiments", "results")
CHANNELS = ["clean", "noise_20db", "noise_10db"]


def _apply_channel_mac(wav, name):
    rng = np.random.default_rng(SEED)
    if name == "clean":
        return np.asarray(wav, np.float64)
    if name == "noise_20db":
        return _add_noise(wav, 20.0, rng)
    if name == "noise_10db":
        return _add_noise(wav, 10.0, rng)
    raise KeyError(name)


def main():
    import jiwer
    from src.content.annotation import Annotation
    from src.content.detect import detect_sensitive
    from src.content.encrypt import decrypt_spans, encrypt_spans, gen_key
    from src.content.sensitive_lists import SensitiveLists
    from experiments.eval_asr import load_asr

    print("== A2 采样（确定性重放） ==", flush=True)
    sl = SensitiveLists.build()
    a2 = pick_a2_files(sl)

    # 构造密文 + MAC（明文帧体哈希，随密钥包分发）
    items = []
    for f, wav, ctc, hits in a2:
        ann = Annotation.from_ctc(f, ctc)
        by_idx = {h.idx: h for h in hits}
        idxs = sorted(h.idx for h in hits)
        groups, cur = [], [idxs[0]]
        for j in idxs[1:]:
            if j == cur[-1] + 1:
                cur.append(j)
            else:
                groups.append(cur)
                cur = [j]
        groups.append(cur)
        for g in groups:
            ann.mark_words(g, by_idx[g[0]].sources[0])
        spans = ann.merge_spans(margin_frames=8)
        keys = [gen_key() for _ in spans]
        pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
        enc = encrypt_spans(pcm, spans, keys)
        tags = []
        for fs, fe in spans:
            body = pcm[fs * 576:(fe + 1) * 576].tobytes()
            tags.append(hashlib.sha256(body).digest()[:4])
        words = {h.word.upper().strip(".,!?;:") for h in hits}
        items.append({"file": f, "pcm": pcm, "spans": spans, "keys": keys,
                      "tags": tags, "enc": enc, "words": words})
    print(f"{len(items)} 句 / {sum(len(it['words']) for it in items)} 敏感词", flush=True)

    def decrypt_mac(pcm_enc, spans, keys, tags):
        """MAC 校验解密：校验失败区间回退静音。"""
        out = decrypt_spans(pcm_enc, spans, keys)
        for j, (fs, fe) in enumerate(spans):
            body = out[fs * 576:(fe + 1) * 576].tobytes()
            if hashlib.sha256(body).digest()[:4] != tags[j]:
                out[fs * 576:(fe + 1) * 576] = 0      # 干净失败：静音
        return out

    # 逐字节自检：clean 信道往返必须与原始 PCM 精确一致（MAC 语义的前提）
    bad = sum(1 for it in items
              if not np.array_equal(
                  decrypt_spans(
                      np.clip(np.round(_apply_channel_mac(
                          it["enc"].astype(np.float32) / 32768.0, "clean")
                          * 32768.0), -32768, 32767).astype(np.int16),
                      it["spans"], it["keys"]), it["pcm"]))
    print(f"clean 往返逐字节一致：{len(items) - bad}/{len(items)} 句", flush=True)
    assert bad == 0, "clean 往返不精确，MAC 语义不成立"

    model = load_asr(device="cuda:0", overrides="test_beam_size: 4")
    cache = json.load(open(HYP_CACHE, encoding="utf-8")) if os.path.exists(HYP_CACHE) else {}

    # 注意：float↔int16 往返必须逐字节精确（正向 /32768、逆向 round(*32768)），
    # 否则"clean 信道"本身就有 1-LSB 扰动，严格字节 MAC 会把合法解密也全部拒绝
    # （exp11 首跑实测踩中：/32768 配 *32767 截断 → clean mac FNR 0.983=全部拒解）。
    jobs = []
    for ch in CHANNELS:
        for i, it in enumerate(items):
            chan = _apply_channel_mac(
                it["enc"].astype(np.float32) / 32768.0, ch)
            pcm_ch = np.clip(np.round(chan * 32768.0),
                             -32768, 32767).astype(np.int16)
            naive = decrypt_spans(pcm_ch, it["spans"], it["keys"])
            mac = decrypt_mac(pcm_ch, it["spans"], it["keys"], it["tags"])
            jobs.append((f"X11b|{ch}|naive|{i}", naive.astype(np.float32) / 32768.0))
            jobs.append((f"X11b|{ch}|mac|{i}", mac.astype(np.float32) / 32768.0))
    transcribe_cached(model, jobs, cache)

    out = {"protocol": {"channels": CHANNELS, "margin_frames": 8,
                        "mac": "SHA256(明文)[:4] per span, fail->silence",
                        "asr": "Conformer beam=4",
                        "fix_note": "v2：float->int16 改 round(*32768) 保证 clean 逐字节精确"
                                    "（v1 用 *32767 截断，1-LSB 扰动令 MAC 连合法解密也全拒）"},
           "fnr": {}}
    for ch in CHANNELS:
        row = {}
        for variant in ("naive", "mac"):
            tot = rec = 0
            survivors = []
            for i, it in enumerate(items):
                hyp = set(normalize(cache[f"X11b|{ch}|{variant}|{i}"]).split())
                for w in it["words"]:
                    tot += 1
                    if w in hyp:
                        rec += 1
                        survivors.append(w)
            row[variant] = {"total": tot, "survived": rec,
                            "fnr": round(1 - rec / tot, 4),
                            "survivors_sample": survivors[:10]}
        out["fnr"][ch] = row
        print(f"{ch}: naive FNR={row['naive']['fnr']:.3f} "
              f"({row['naive']['survived']} 漏)  mac FNR={row['mac']['fnr']:.3f} "
              f"({row['mac']['survived']} 漏)", flush=True)

    # ── E：quiet 版窗内核心区静音度复测（排除 ±100ms 边界 slop） ──
    print("== E quiet 窗核心复测 ==", flush=True)
    from src.content.mp3_crypto import encode_mp3, parse, quiet_mp3, time_spans_to_frames
    from src.ffmpeg_util import find_ffmpeg
    ff = find_ffmpeg()
    probe = os.path.join(PROJ, "data", "probe", "exp11")
    os.makedirs(probe, exist_ok=True)
    rows = []
    for it in items[:8]:
        spans_sec = [(fs * 576 / SR, (fe + 1) * 576 / SR) for fs, fe in it["spans"]]
        tmp = os.path.join(probe, "t.wav")
        sf.write(tmp, it["enc"].astype(np.float32) / 32768.0, SR)
        mp3 = os.path.join(probe, "t.mp3")
        encode_mp3(tmp, mp3, kbps=128)
        data = open(mp3, "rb").read()
        ranges = time_spans_to_frames(parse(data), spans_sec)
        open(os.path.join(probe, "q.mp3"), "wb").write(quiet_mp3(data, ranges))
        subprocess.run([ff, "-y", "-v", "error", "-i", os.path.join(probe, "q.mp3"),
                        "-ar", "44100", "-ac", "1", "-b:a", "64k",
                        os.path.join(probe, "q64.mp3")], check=True)
        dec = os.path.join(probe, "q64.wav")
        subprocess.run([ff, "-y", "-v", "error", "-i", os.path.join(probe, "q64.mp3"),
                        "-ar", str(SR), "-ac", "1", dec], check=True)
        y, _ = sf.read(dec)
        y = np.asarray(y, np.float64)
        core = []
        for a, b in spans_sec:
            a2, b2 = int((a + 0.1) * SR), int((b - 0.1) * SR)
            if b2 > a2 and b2 <= len(y):
                core.append(float(np.sqrt((y[a2:b2] ** 2).mean())))
        if core:
            rows.append({"file": os.path.basename(it["file"]),
                         "core_rms_db": round(20 * np.log10(min(core) + 1e-12), 1)})
    out["quiet_core_silence"] = rows
    print(json.dumps(rows, ensure_ascii=False), flush=True)

    json.dump(out, open(os.path.join(RES, "exp11_mac_integrity.json"), "w",
                        encoding="utf-8"), indent=1, ensure_ascii=False)
    print("-> experiments/results/exp11_mac_integrity.json", flush=True)


if __name__ == "__main__":
    main()
