"""[M7 加分项 ①②] 真实信道鲁棒性：问题发现 + 控制变量扫描 harness。

背景（执行计划 M7 加分项）：SpeechGuard 的两大承诺——持密钥可精确恢复、敏感词
加密不外泄——都建立在"发布音频原样送达"的假设上。真实信道（微信语音重编码、
电话 8k、扬声器→麦克风翻录、噪声）会破坏这个假设。本脚本量化破坏程度：

  A1 声学恢复链：protect → 信道 → recover → 测「恢复后 WER」与「说话人相似度」
     （论文口径的"恢复质量"在信道下是否还成立）
  A2 内容加密链：encrypt(敏感帧) → 信道 → 按原帧号 decrypt → 测「敏感词幸存率
     （FNR）」与「对齐漂移 ms」
     （帧网格被信道挪动后，加密区间还盖得住吗？管理员还解得开吗？）

信道实现（全部本地可复现，无外部语料）：
  clean        直通（基线）
  mp3_128k     16k→44.1k MP3(128k)→16k（"微信语音"代理：计划允许的低码率重编码模拟）
  mp3_64k      同上 64kbps（②档扫描点）
  resample_8k  16k→8k→16k（电话带宽；帧网格仍整除，作为"纯带宽损伤"对照）
  noise_10db   加性白噪 SNR=10dB
  rerecord     合成房间 IR 卷积 + 150-6500Hz 带通 + 30dB 底噪（翻录仿真）
  trim_400ms   VAD 掐头 400ms（内容相对帧网格平移，超过 288ms 加密余量→边缘渗漏）

协议对齐：single 模式 dist∈(0.5,0.6) n∈[1,10]；检测在原始音频（与 pipeline/eval_fnr
一致）；margin=8 帧；评估 ASR=Conformer beam=4；ASV=ECAPA 余弦。样本 test-clean
10 人×5 句（A2 用其中含≥2敏感词的 50 句，采样法同 eval_fnr）。

运行（GPU，约 30-45 分钟，转写结果磁盘缓存可断点续跑）：
  E:/Miniconda/envs/speechguard/python.exe experiments/exp8_channel_cases.py
输出：experiments/results/exp8_cases.json + exp8_cases.md + data/probe/exp8/ 试听样例
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfiltfilt

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.eval_asr import librispeech_ref, normalize  # noqa: E402
from src.ffmpeg_util import find_ffmpeg  # noqa: E402

SR = 16000
DEVICE = "cuda:0"
RES = os.path.join(PROJ, "experiments", "results")
PROBE = os.path.join(PROJ, "data", "probe", "exp8")
HYP_CACHE = os.path.join(RES, "exp8_hyp_cache.json")

N_SPEAKERS, N_PER_SPK = 10, 5          # A1 样本
N_FN = 50                               # A2 样本（含≥2敏感词）
MARGIN = 8                              # 与 M4 定稿一致
SEED = 808

CHANNELS = ["clean", "mp3_128k", "mp3_64k", "resample_8k", "noise_10db",
            "rerecord", "trim_400ms"]


# ─────────────────────────────── 信道实现 ───────────────────────────────

def _mp3_roundtrip(wav: np.ndarray, kbps: int) -> np.ndarray:
    ff = find_ffmpeg()
    with tempfile.TemporaryDirectory() as td:
        w16, m, w2 = (os.path.join(td, n) for n in ("a.wav", "a.mp3", "b.wav"))
        sf.write(w16, wav, SR)
        subprocess.run([ff, "-y", "-v", "error", "-i", w16, "-ar", "44100",
                        "-ac", "1", "-b:a", f"{kbps}k", m], check=True)
        subprocess.run([ff, "-y", "-v", "error", "-i", m, "-ar", str(SR),
                        "-ac", "1", w2], check=True)
        out, _ = sf.read(w2)
    out = np.asarray(out, np.float64)
    n = len(wav)
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return out[:n]


def _resample(wav, sr_from, sr_to):
    import librosa
    return librosa.resample(np.asarray(wav, np.float64),
                            orig_sr=sr_from, target_sr=sr_to)


def _bandpass(wav, lo, hi, order=4):
    sos = butter(order, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, np.asarray(wav, np.float64))


def _add_noise(wav, snr_db, rng):
    rms = np.sqrt(np.mean(wav ** 2)) + 1e-12
    sigma = rms / (10 ** (snr_db / 20))
    return wav + rng.normal(0.0, sigma, len(wav))


def _rerecord(wav, rng):
    """扬声器→麦克风翻录仿真：合成指数衰减噪声 IR（RT60≈0.35s）卷积
    + 150-6500Hz 带通（小喇叭带宽）+ 30dB 底噪。"""
    ir_len = int(0.35 * SR)
    ir = rng.normal(0, 1, ir_len) * np.exp(-np.linspace(0, 6.9, ir_len))
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir ** 2))
    y = fftconvolve(wav, ir)[: len(wav)]
    y = _bandpass(y, 150, 6500)
    y = _add_noise(y, 30.0, rng)
    peak = np.abs(y).max()
    if peak > 0.98:
        y *= 0.98 / peak
    return y


def _trim(wav, ms):
    """VAD 掐头：内容左移 ms 毫秒（微信语音等会裁掉头部静音/弱音，
    使内容相对帧网格平移——平移量超过加密余量即产生边缘渗漏）。"""
    k = int(ms / 1000 * SR)
    y = np.zeros_like(wav)
    y[: len(wav) - k] = wav[k:]
    return y


def apply_channel(name: str, wav: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)          # 噪声/IR 固定流：可复现
    if name == "clean":
        return np.asarray(wav, np.float64)
    if name == "mp3_128k":
        return _mp3_roundtrip(wav, 128)
    if name == "mp3_64k":
        return _mp3_roundtrip(wav, 64)
    if name == "resample_8k":
        return _resample(_resample(wav, SR, 8000), 8000, SR)
    if name == "noise_10db":
        return _add_noise(wav, 10.0, rng)
    if name == "rerecord":
        return _rerecord(wav, rng)
    if name == "trim_400ms":
        return _trim(wav, 400)
    raise KeyError(name)


def align_drms(a: np.ndarray, b: np.ndarray, max_ms: float = 600.0) -> float:
    """b 相对 a 的整体时移（ms，正值 = b 滞后于 a）。FFT 互相关取峰。"""
    n = len(a)
    f = np.conj(np.fft.rfft(a - a.mean(), 2 * n)) * np.fft.rfft(b - b.mean(), 2 * n)
    corr = np.fft.irfft(f, 2 * n)
    lag = int(np.argmax(corr))
    if lag > n:
        lag -= 2 * n
    lim = max_ms * SR / 1000
    return float(lag * 1000 / SR) if abs(lag) <= lim else float("nan")


# ─────────────────────────────── 数据采样 ───────────────────────────────

def pick_a1_files():
    files = sorted(glob.glob(os.path.join(
        PROJ, "data", "librispeech", "LibriSpeech", "test-clean", "*", "*", "*.flac")))
    by_spk: dict[str, list] = {}
    for f in files:
        by_spk.setdefault(f.split(os.sep)[-3], []).append(f)
    spks = sorted(by_spk)[:N_SPEAKERS]
    out = []
    for s in spks:
        for f in by_spk[s][:N_PER_SPK]:
            out.append((f, s))
    return out


def pick_a2_files(sl):
    """同 eval_fnr：顺序扫描直到凑足 N_FN 条含 ≥2 敏感词的语音。"""
    from src.content.detect import detect_sensitive
    from src.content.transcribe import transcribe_with_times

    files = sorted(glob.glob(os.path.join(
        PROJ, "data", "librispeech", "LibriSpeech", "test-clean", "*", "*", "*.flac")))
    picked, scanned = [], 0
    for f in files:
        wav, _ = sf.read(f)
        ctc = transcribe_with_times(wav.astype(np.float32), device=DEVICE)
        hits = detect_sensitive(ctc, sl, device=DEVICE)
        scanned += 1
        if len(hits) >= 2:
            picked.append((f, wav.astype(np.float64), ctc, hits))
            if len(picked) >= N_FN:
                break
    print(f"A2 采样：扫描 {scanned} 句 → {len(picked)} 句，"
          f"敏感词 {sum(len(h) for *_, h in picked)} 个", flush=True)
    return picked


# ─────────────────────────────── 转写缓存 ───────────────────────────────

def transcribe_cached(model, jobs, cache):
    """jobs: [(key, wav)] → {key: hyp}，磁盘缓存断点续跑。"""
    todo = [(k, w) for k, w in jobs if k not in cache]
    print(f"  缓存命中 {len(jobs) - len(todo)}/{len(jobs)}，待转 {len(todo)}", flush=True)
    from experiments.eval_asr import transcribe
    t0 = time.time()
    done = 0
    for s in range(0, len(todo), 30):
        chunk = todo[s:s + 30]
        hyps = transcribe(model, [w for _, w in chunk], batch_size=2)
        for (k, _), h in zip(chunk, hyps):
            cache[k] = h
        done += len(chunk)
        el = time.time() - t0
        print(f"  转写 {done}/{len(todo)}（{el:.0f}s，"
              f"剩 ~{el / done * (len(todo) - done):.0f}s）", flush=True)
        json.dump(cache, open(HYP_CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(cache, open(HYP_CACHE, "w", encoding="utf-8"), ensure_ascii=False)


# ─────────────────────────────── 主流程 ───────────────────────────────

def main():
    os.makedirs(PROBE, exist_ok=True)
    import jiwer
    import torch
    from experiments.eval_asv import embed_wavs, load_asv
    from experiments.eval_asr import load_asr
    from src.acoustic.warping import protect, recover
    from src.content.annotation import Annotation
    from src.content.encrypt import decrypt_spans, encrypt_spans, gen_key
    from src.content.sensitive_lists import SensitiveLists

    rng = np.random.default_rng(SEED)

    # ── A1：声学恢复链 ──
    print("== A1 样本加载 + protect ==", flush=True)
    a1 = []
    for f, spk in pick_a1_files():
        wav, _ = sf.read(f)
        prot, arch = protect(np.asarray(wav, np.float64), rng=rng)
        a1.append({"file": os.path.relpath(f, PROJ), "spk": spk, "ref": librispeech_ref(f),
                   "wav": wav, "prot": prot, "arch": arch})
    print(f"A1：{len(a1)} 句 / {len({x['spk'] for x in a1})} 说话人", flush=True)

    asv = load_asv(device=DEVICE)
    cache = json.load(open(HYP_CACHE, encoding="utf-8")) if os.path.exists(HYP_CACHE) else {}

    # 信道输出缓存：每 (句, 信道) 只算一次（resample/mp3 不便宜）
    print("== A1 信道施加（缓存） ==", flush=True)
    _librosa_warm = _resample(np.zeros(SR // 2, np.float64), SR, 8000)  # 预热
    ch_out = {(i, ch): apply_channel(ch, x["prot"])
              for i, x in enumerate(a1) for ch in CHANNELS}
    rec_out = {(i, ch): recover(ch_out[(i, ch)], a1[i]["arch"])
               for i in range(len(a1)) for ch in CHANNELS}

    # 嵌入：original / protected / 各信道 recover → 说话人相似度
    emb_jobs = {"original": [x["wav"] for x in a1],
                "protected": [x["prot"] for x in a1]}
    for ch in CHANNELS:
        emb_jobs[f"recov_{ch}"] = [rec_out[(i, ch)] for i in range(len(a1))]
    embs = {k: embed_wavs(asv, [np.asarray(w, np.float32) for w in ws])
            for k, ws in emb_jobs.items()}
    sim = {k: float(np.mean(np.sum(embs["original"] * v, axis=1)))
           for k, v in embs.items() if k != "original"}

    # ASR：recover 后 + protected 直接信道后（攻击者视角可懂度）
    print("== A1 ASR ==", flush=True)
    asr_jobs = []
    for ch in CHANNELS:
        for i in range(len(a1)):
            asr_jobs.append((f"A1|{ch}|recov|{i}", rec_out[(i, ch)]))
            asr_jobs.append((f"A1|{ch}|prot|{i}", ch_out[(i, ch)]))
    model = load_asr(device=DEVICE, overrides="test_beam_size: 4")
    transcribe_cached(model, asr_jobs, cache)

    def wer_of(ch, kind):
        errs = tot = 0
        for i, x in enumerate(a1):
            h = normalize(cache[f"A1|{ch}|{kind}|{i}"])
            r = normalize(x["ref"])
            w = jiwer.wer(r, h)
            errs += round(w * len(r.split())); tot += len(r.split())
        return round(100 * errs / tot, 2)

    a1_res = {"n_utts": len(a1), "n_speakers": len({x["spk"] for x in a1}),
              "sim_to_original": {k: round(v, 3) for k, v in sim.items()},
              "wer_recovered": {ch: wer_of(ch, "recov") for ch in CHANNELS},
              "wer_protected": {ch: wer_of(ch, "prot") for ch in CHANNELS}}
    print("A1 WER(recov):", a1_res["wer_recovered"], flush=True)
    print("A1 WER(prot) :", a1_res["wer_protected"], flush=True)
    print("A1 sim(orig) :", a1_res["sim_to_original"], flush=True)

    # 试听样例（前 3 句 × 关键信道）
    for i in range(3):
        x = a1[i]
        sf.write(os.path.join(PROBE, f"case{i}_original.wav"), x["wav"], SR)
        sf.write(os.path.join(PROBE, f"case{i}_protected.wav"), x["prot"], SR)
        sf.write(os.path.join(PROBE, f"case{i}_recov_mp3_64k.wav"),
                 rec_out[(i, "mp3_64k")], SR)
        sf.write(os.path.join(PROBE, f"case{i}_recov_rerecord.wav"),
                 rec_out[(i, "rerecord")], SR)

    # ── A2：内容加密链 ──
    print("== A2 采样 + 加密 ==", flush=True)
    sl = SensitiveLists.build()
    a2 = pick_a2_files(sl)
    a2_res_items = []
    jobs = []
    for i, (f, wav, ctc, hits) in enumerate(a2):
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
        spans = ann.merge_spans(margin_frames=MARGIN)
        keys = [gen_key() for _ in spans]
        pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
        enc = encrypt_spans(pcm, spans, keys)
        words = {h.word.upper().strip(".,!?;:") for h in hits}
        a2_res_items.append({"file": os.path.relpath(f, PROJ), "ref": librispeech_ref(f),
                             "words": sorted(words), "n_spans": len(spans)})
        for ch in CHANNELS:
            chan = apply_channel(ch, enc.astype(np.float32) / 32768.0)
            dec = decrypt_spans((np.clip(chan, -1, 1) * 32767).astype(np.int16),
                                spans, keys).astype(np.float32) / 32768.0
            jobs.append((f"A2|{ch}|dec|{i}", dec))
            if ch == "mp3_64k":
                jobs.append((f"A2|{ch}|nod|{i}", chan.astype(np.float32)))
            drift = align_drms(wav, dec)
            a2_res_items[i].setdefault("drift_ms", {})[ch] = round(drift, 1)

    print("== A2 ASR ==", flush=True)
    transcribe_cached(model, jobs, cache)

    fnr = {}
    for ch in CHANNELS:
        tot = rec = 0
        survivors = []
        for i, it in enumerate(a2_res_items):
            hyp_tokens = set(normalize(cache[f"A2|{ch}|dec|{i}"]).split())
            for w in it["words"]:
                tot += 1
                if w in hyp_tokens:
                    rec += 1
                    survivors.append({"utt": it["file"], "word": w})
        fnr[ch] = {"total": tot, "survived": rec, "fnr": round(1 - rec / tot, 4),
                   "survivors": survivors[:15]}
    # 攻击者视角（密文不解密）FNR——仅当 nod 任务存在时评分
    fnr_nod = {}
    for ch in CHANNELS:
        tot = rec = 0
        for i, it in enumerate(a2_res_items):
            k = f"A2|{ch}|nod|{i}"
            if k not in cache:
                continue
            hyp_tokens = set(normalize(cache[k]).split())
            for w in it["words"]:
                tot += 1
                if w in hyp_tokens:
                    rec += 1
        if tot:
            fnr_nod[ch] = {"total": tot, "survived": rec,
                           "fnr": round(1 - rec / tot, 4)}

    a2_res = {"n_utts": len(a2_res_items),
              "margin_frames": MARGIN,
              "utt_words": [{"file": it["file"], "words": it["words"]}
                            for it in a2_res_items],
              "fnr_attacker_no_decrypt": fnr_nod,
              "fnr_after_decrypt": fnr,
              "drift_ms": {it["file"]: it["drift_ms"] for it in a2_res_items[:5]},
              "drift_median_ms": {
                  ch: round(float(np.median([it["drift_ms"][ch] for it in a2_res_items])), 1)
                  for ch in CHANNELS}}
    print("A2 FNR(dec|ch):", {k: v["fnr"] for k, v in fnr.items()}, flush=True)
    print("A2 drift median ms:", a2_res["drift_median_ms"], flush=True)

    # ── 落盘 ─────────────────────────────────────────────────────────
    out = {"protocol": {
              "dataset": "LibriSpeech test-clean", "seed": SEED,
              "warping": "single dist(0.5,0.6) n(1,10)", "margin_frames": MARGIN,
              "asr": "Conformer beam=4", "asv": "ECAPA cosine",
              "channels": CHANNELS,
              "note": "mp3 为计划允许的低码率重编码代理（微信语音 Silk/AMR 无编码器可用）"},
        "A1": a1_res, "A2": a2_res}
    json.dump(out, open(os.path.join(RES, "exp8_cases.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("-> experiments/results/exp8_cases.json", flush=True)
    print("-> data/probe/exp8/ 试听样例", flush=True)


if __name__ == "__main__":
    main()
