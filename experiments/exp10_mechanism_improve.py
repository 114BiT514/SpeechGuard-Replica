"""[M7 加分项 ③④] 机理验证 + 改进方案对照实验。

四个部分（结果合并写进 exp10_mechanism_improve.json）：

A 机理·时移错配（验证 exp8-F3 推论）：
   warping 模式 {single, ola, block} × 信道 {clean, trim_400ms}
   预测：single 整句统一映射对全局时移免疫；ola/block 逐段映射在时移后
   "帧位置↔映射段"错配，恢复崩坏。若证实，反证 M3 选 single 的额外鲁棒收益。

B 机理·频带证据（解释 exp8-F1/F2）：
   各信道 recover 后音频的分带能量损失表（0-2k/2-4k/4-8k）+ 与原始的分带谱相关。
   预测：8k 信道抹掉 4-8k 带（声纹主战场）→ sim 0.66；mp3_64k 三带俱在 → sim 0.90。

C 改进·同步锚点预对齐（针对 F4/trim，执行计划"恢复端先均衡再逆扭曲"的实例化）：
   protect 端把"首个有声帧位置"写入密钥包（S 的合法扩展，不泄露内容语义之外的信息）；
   恢复端 VAD 检测收到的音频首有声帧 → 回滚时移 → 再逆扭曲。
   对照：trim/noise/rerecord/8k 四信道 × {原始恢复, 预对齐恢复} 的 WER/sim。

D 改进·quiet 分发版信道存活（针对 F4 的可用性兜底）：
   quiet MP3 → mp3_64k 二次转码 → 解码：验证时长保持、敏感窗仍静音、
   窗外内容可懂度（谱相关）。证明"时间级恢复"语义天然免疫字节级信道破坏。

运行（GPU ~20-30 分钟，ASR 走 exp8 缓存命名空间）：
  E:/Miniconda/envs/speechguard/python.exe experiments/exp10_mechanism_improve.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.eval_asr import librispeech_ref, normalize  # noqa: E402
from experiments.exp8_channel_cases import (  # noqa: E402
    HYP_CACHE, SEED, SR, _resample, apply_channel, pick_a1_files,
    transcribe_cached)

RES = os.path.join(PROJ, "experiments", "results")
BANDS = [(0, 2000), (2000, 4000), (4000, 8000)]


def first_voiced_frame(wav, nf_db=None, hop=576):
    """能量法首有声帧（576 网格）。阈值：P95-25dB（对数字静音门控安全）。"""
    n = len(wav) // hop
    if n == 0:
        return 0
    db = 20 * np.log10(np.sqrt((wav[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-12)
    thr = float(np.percentile(db, 95)) - 25.0
    idx = np.where(db >= thr)[0]
    return int(idx[0]) if len(idx) else 0


def band_stats(orig, rec):
    """分带能量损失 dB + 谱相关。"""
    n = min(len(orig), len(rec))
    O = np.abs(np.fft.rfft(orig[:n] - orig[:n].mean()))
    R = np.abs(np.fft.rfft(rec[:n] - rec[:n].mean()))
    fr = np.fft.rfftfreq(n, 1 / SR)
    out = {}
    for lo, hi in BANDS:
        m = (fr >= lo) & (fr < hi)
        e_o, e_r = float((O[m] ** 2).mean()), float((R[m] ** 2).mean())
        loss = 10 * np.log10((e_r + 1e-18) / (e_o + 1e-18))
        cc = float(np.corrcoef(O[m], R[m])[0, 1]) if m.sum() > 8 else float("nan")
        out[f"{lo // 1000}-{hi // 1000}k"] = {"loss_db": round(float(loss), 1),
                                              "spec_corr": round(cc, 3)}
    return out


def main():
    import jiwer
    from experiments.eval_asv import embed_wavs, load_asv
    from experiments.eval_asr import load_asr
    from src.acoustic.warping import protect, recover

    rng = np.random.default_rng(SEED)
    print("== 样本 + 三模式 protect ==", flush=True)
    a1 = []
    for f, spk in pick_a1_files():
        wav, _ = sf.read(f)
        wav = np.asarray(wav, np.float64)
        rec = {"file": f, "spk": spk, "ref": librispeech_ref(f), "wav": wav}
        for mode in ("single", "ola", "block"):
            p, arch = protect(wav, rng=np.random.default_rng(SEED + 7), mode=mode)
            rec[mode] = (p, arch)
        a1.append(rec)
    print(f"{len(a1)} 句", flush=True)

    asv = load_asv(device="cuda:0")
    model = load_asr(device="cuda:0", overrides="test_beam_size: 4")
    cache = json.load(open(HYP_CACHE, encoding="utf-8")) if os.path.exists(HYP_CACHE) else {}
    out = {}

    # ── A：模式 × 时移 错配矩阵 ────────────────────────────────────
    print("== A 机理：模式×trim ==", flush=True)
    jobsA, recA = [], {}
    for mode in ("single", "ola", "block"):
        for ch in ("clean", "trim_400ms"):
            rs = [recover(apply_channel(ch, x[mode][0]), x[mode][1], mode=mode)
                  for x in a1]
            recA[(mode, ch)] = rs
            e_org = embed_wavs(asv, [np.asarray(x["wav"], np.float32) for x in a1])
            e_rec = embed_wavs(asv, [np.asarray(r, np.float32) for r in rs])
            sim = float(np.mean(np.sum(e_org * e_rec, axis=1)))
            out[f"A|{mode}|{ch}|sim"] = round(sim, 3)
            for i, r in enumerate(rs):
                jobsA.append((f"X10|A|{mode}|{ch}|{i}", r))
    transcribe_cached(model, jobsA, cache)
    for mode in ("single", "ola", "block"):
        for ch in ("clean", "trim_400ms"):
            errs = tot = 0
            for i, x in enumerate(a1):
                h = normalize(cache[f"X10|A|{mode}|{ch}|{i}"])
                r = normalize(x["ref"])
                errs += round(jiwer.wer(r, h) * len(r.split()))
                tot += len(r.split())
            out[f"A|{mode}|{ch}|wer"] = round(100 * errs / tot, 2)
    print({k: v for k, v in out.items() if k.startswith("A|")}, flush=True)

    # ── B：分带频域证据 ───────────────────────────────────────────
    print("== B 机理：分带能量 ==", flush=True)
    bands = {}
    for ch in ("mp3_64k", "mp3_128k", "resample_8k", "noise_10db", "rerecord"):
        st = []
        for x in a1[:10]:
            r = recover(apply_channel(ch, x["single"][0]), x["single"][1])
            st.append(band_stats(x["wav"], r))
        agg = {b: {"loss_db": round(float(np.mean([s[b]["loss_db"] for s in st])), 1),
                   "spec_corr": round(float(np.mean([s[b]["spec_corr"] for s in st])), 3)}
               for b in ("0-2k", "2-4k", "4-8k")}
        bands[ch] = agg
    out["B_band_evidence"] = bands
    print(json.dumps(bands, ensure_ascii=False), flush=True)

    # ── C：同步锚点预对齐 ─────────────────────────────────────────
    print("== C 改进：预对齐（含同种子未对齐基线） ==", flush=True)
    jobsC, recC = [], {}
    for ch in ("trim_400ms", "noise_10db", "rerecord", "resample_8k"):
        naive = [recover(apply_channel(ch, x["single"][0]), x["single"][1])
                 for x in a1]
        e_org0 = embed_wavs(asv, [np.asarray(x["wav"], np.float32) for x in a1])
        out[f"C|{ch}|sim_naive"] = round(
            float(np.mean(np.sum(e_org0 * embed_wavs(
                asv, [np.asarray(r, np.float32) for r in naive]), axis=1))), 3)
        for i, r in enumerate(naive):
            jobsC.append((f"X10|Cbase|{ch}|{i}", r))
        aligned = []
        for x in a1:
            prot = x["single"][0]
            anchor = first_voiced_frame(prot)                 # protect 端写入密钥包
            recv = apply_channel(ch, prot)
            got = first_voiced_frame(recv)
            d = got - anchor                                   # 帧差，负=内容前移
            if abs(d) > 50:                                    # >1.8s 视为 VAD 误检，不对齐
                d = 0
            k = int(d) * 576                                   # 帧差→样本移位
            y = np.zeros_like(recv)
            if k <= 0:
                y[-k:] = recv[: len(recv) + k]
            else:
                y[: len(y) - k] = recv[k:]
            aligned.append(recover(y, x["single"][1]))
        recC[ch] = aligned
        e_org = embed_wavs(asv, [np.asarray(x["wav"], np.float32) for x in a1])
        e_rec = embed_wavs(asv, [np.asarray(r, np.float32) for r in aligned])
        out[f"C|{ch}|sim_aligned"] = round(
            float(np.mean(np.sum(e_org * e_rec, axis=1))), 3)
        for i, r in enumerate(aligned):
            jobsC.append((f"X10|C|{ch}|{i}", r))
    transcribe_cached(model, jobsC, cache)
    for ch in ("trim_400ms", "noise_10db", "rerecord", "resample_8k"):
        for tag, kp in (("naive", "X10|Cbase|{}|{}"), ("aligned", "X10|C|{}|{}")):
            errs = tot = 0
            for i, x in enumerate(a1):
                h = normalize(cache[kp.format(ch, i)])
                r = normalize(x["ref"])
                errs += round(jiwer.wer(r, h) * len(r.split()))
                tot += len(r.split())
            out[f"C|{ch}|wer_{tag}"] = round(100 * errs / tot, 2)
    print({k: v for k, v in out.items() if k.startswith("C|")}, flush=True)

    # ── D：quiet 分发版信道存活 ───────────────────────────────────
    print("== D 改进：quiet 版存活 ==", flush=True)
    from src.content.annotation import Annotation
    from src.content.encrypt import encrypt_spans, gen_key
    from src.content.mp3_crypto import encode_mp3, parse, quiet_mp3, time_spans_to_frames
    from src.content.sensitive_lists import SensitiveLists
    from src.content.transcribe import transcribe_with_times
    from src.content.detect import detect_sensitive

    sl = SensitiveLists.build()
    d_rows = []
    for x in a1[:8]:
        ctc = transcribe_with_times(x["wav"].astype(np.float32), device="cuda:0")
        hits = detect_sensitive(ctc, sl, device="cuda:0")
        if not hits:
            continue
        ann = Annotation.from_ctc(x["file"], ctc)
        for h in hits:
            ann.mark_words([h.idx], h.sources[0])
        spans = ann.merge_spans(margin_frames=8)
        spans_sec = [(fs * 576 / SR, (fe + 1) * 576 / SR) for fs, fe in spans]
        pcm = (np.clip(x["single"][0], -1, 1) * 32767).astype(np.int16)
        keys = [gen_key() for _ in spans]
        enc = encrypt_spans(pcm, spans, keys)
        tmp = os.path.join(PROJ, "data", "probe", "exp10_tmp.wav")
        sf.write(tmp, enc.astype(np.float32) / 32768.0, SR)
        mp3 = tmp.replace(".wav", ".mp3")
        encode_mp3(tmp, mp3, kbps=128)
        data = open(mp3, "rb").read()
        ranges = time_spans_to_frames(parse(data), spans_sec)
        quiet = quiet_mp3(data, ranges)
        # 二次转码（模拟微信再压缩）
        q2 = os.path.join(PROJ, "data", "probe", "exp10_q2.wav")
        sf.write(q2, np.zeros(SR), SR)
        import subprocess
        from src.ffmpeg_util import find_ffmpeg
        ff = find_ffmpeg()
        with open(os.path.join(PROJ, "data", "probe", "exp10_q.mp3"), "wb") as fh:
            fh.write(quiet)
        subprocess.run([ff, "-y", "-v", "error", "-i",
                        os.path.join(PROJ, "data", "probe", "exp10_q.mp3"),
                        "-ar", "44100", "-ac", "1", "-b:a", "64k",
                        os.path.join(PROJ, "data", "probe", "exp10_q64.mp3")], check=True)
        subprocess.run([ff, "-y", "-v", "error", "-i",
                        os.path.join(PROJ, "data", "probe", "exp10_q64.mp3"),
                        "-ar", str(SR), "-ac", "1", q2], check=True)
        y, _ = sf.read(q2)
        y = np.asarray(y, np.float64)
        # 窗内静音度 + 窗外与原始（保护态）的相关
        win_rms = max((float(np.sqrt((y[int(a * SR):int(b * SR)] ** 2).mean()))
                       for a, b in spans_sec if b * SR <= len(y)), default=float("nan"))
        out_mask = np.ones(len(y), bool)
        for a, b in spans_sec:
            out_mask[int(a * SR):int(b * SR)] = False
        n = min(len(y), len(x["single"][0]))
        cc_out = float(np.corrcoef(y[:n][out_mask[:n]],
                                   x["single"][0][:n][out_mask[:n]])[0, 1])
        d_rows.append({"file": os.path.basename(x["file"]),
                       "n_spans": len(spans),
                       "window_rms_db": round(20 * np.log10(win_rms + 1e-12), 1),
                       "outside_spec_corr": round(cc_out, 3)})
    out["D_quiet_survival"] = d_rows
    print(json.dumps(d_rows, ensure_ascii=False), flush=True)

    json.dump(out, open(os.path.join(RES, "exp10_mechanism_improve.json"), "w",
                        encoding="utf-8"), indent=1, ensure_ascii=False)
    print("-> experiments/results/exp10_mechanism_improve.json", flush=True)


if __name__ == "__main__":
    main()
