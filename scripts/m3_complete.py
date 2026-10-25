"""[M3 补全] 对齐论文全部未覆盖数字：Table 5 攻击后列 + Table 6 RTC + VCTK 列 + single Fig.7。

顺序执行（单脚本避免 GPU 争抢）：
  1. test-clean（single 模式）：攻击后 EER/WER（reducing 攻击施加于音频）+ RTC
  2. VCTK（40人×65句，44.1k→16k）：原始/保护/攻击 EER + 原始/保护 WER
  3. dev-clean 1000句：single 模式 n=1 dist 十区间扫描（Fig.7 曲线重制）

产出：experiments/results/m3_complete.json + fig_exp1_fig7_single.png
运行：E:/Miniconda/envs/speechguard/python.exe -u scripts/m3_complete.py
"""
import glob
import json
import os
import sys
import time

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

from experiments.eval_asr import corpus_wer, librispeech_ref, load_asr  # noqa: E402
from experiments.eval_asv import dataset_eer, load_asv  # noqa: E402
from src.acoustic.warping import (  # noqa: E402
    FRAME_LEN,
    SegmentParams,
    _apply_single,
    gen_params,
    protect,
    recover,
)

RANGES = [(round(0.1 * i, 1), round(0.1 * (i + 1), 1)) for i in range(10)]
OUT = {}


def attack_audio(prot_wav, rng):
    """reducing 攻击施加于音频：攻击者用随机参数对保护谱再做一次逆扭曲。"""
    a, b, d = gen_params(n=None, rng=rng)
    fake = [SegmentParams(0, max(prot_wav.size // FRAME_LEN, 1), a, b, d)]
    out, _ = _apply_single(prot_wav, archive=fake, invert=True)
    return out


def eval_condition(name, wavs, spk, asv, files=None, refs=None, asr=None, wer_n=300,
                   rng=None, seed_tag=""):
    t0 = time.time()
    eer, _ = dataset_eer(asv, list(zip(wavs, spk)), max_imp_per_utt=1000)
    rec = {"name": name, "eer": round(eer, 4)}
    if asr is not None and refs is not None:
        sel = np.random.default_rng(7).choice(len(wavs), min(wer_n, len(wavs)), replace=False)
        wer, _, _ = corpus_wer(asr, [wavs[i] for i in sel], [refs[i] for i in sel])
        rec["wer"] = round(wer, 4)
    print(f"  [{name}] EER={eer*100:.2f}%" + (f" WER={rec.get('wer', 0)*100:.2f}%" if 'wer' in rec else "")
          + f" ({time.time()-t0:.0f}s)", flush=True)
    return rec


def load_ls(pattern, n_spk=None, n_utt=None):
    dirs = sorted(glob.glob(pattern))
    if n_spk:
        dirs = dirs[:n_spk]
    files = []
    for d in dirs:
        fs = sorted(glob.glob(os.path.join(d, "*.flac")))
        files += fs[:n_utt] if n_utt else fs
    spk = [os.path.basename(os.path.dirname(f)) for f in files]
    refs = [librispeech_ref(f) for f in files]
    wavs = [sf.read(f)[0].astype(np.float64) for f in files]
    return files, spk, refs, wavs


def main():
    t_all = time.time()
    asv = load_asv("cuda:0")
    asr = load_asr("cuda:0", overrides="test_beam_size: 4")
    rng = np.random.default_rng(2026)

    # ---------- 1. test-clean：攻击后 + RTC ----------
    print("== 1. test-clean 攻击后列 + RTC ==", flush=True)
    files, spk, refs, wavs = load_ls(os.path.join(
        PROJ, "data/librispeech/LibriSpeech/test-clean/*/*"))
    prot, archs = [], []
    for w in wavs:
        p, a = protect(w, n=None, dist_range=(0.5, 0.6), rng=rng, mode="single")
        prot.append(p); archs.append(a)
    att = [attack_audio(p, rng) for p in prot]
    OUT["testclean_attacked"] = eval_condition("attacked", att, spk, asv, refs=refs, asr=asr)
    # RTC：protect/recover 耗时 / 音频时长
    t0 = time.time()
    for w in wavs[:200]:
        protect(w, n=None, dist_range=(0.5, 0.6), rng=rng, mode="single")
    t_p = time.time() - t0
    t0 = time.time()
    for w, a in list(zip(prot, archs))[:200]:
        recover(w, a, mode="single")
    t_r = time.time() - t0
    dur = sum(w.size for w in wavs[:200]) / 16000.0
    OUT["rtc"] = {"protect": round(t_p / dur, 4), "recover": round(t_r / dur, 4)}
    print(f"  [RTC] protect={OUT['rtc']['protect']:.3f} recover={OUT['rtc']['recover']:.3f}"
          f" (论文 0.32/0.26)", flush=True)

    # ---------- 2. VCTK 列 ----------
    print("== 2. VCTK（40人×65句）==", flush=True)
    vdirs = sorted(glob.glob(os.path.join(PROJ, "data/vctk/wav48_silence_trimmed/p*")))
    vdirs = [d for d in vdirs if os.path.basename(d) not in ("s5", "log.txt")][:40]
    vfiles, vspk, vrefs = [], [], []
    for d in vdirs:
        fs = sorted(glob.glob(os.path.join(d, "*_mic1.flac")))[:65]
        vfiles += fs
        vspk += [os.path.basename(d)] * len(fs)
    vwav = []
    for f in vfiles:
        w, sr = sf.read(f)
        w = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        vwav.append(w.squeeze().numpy().astype(np.float64))
    # VCTK 转写：txt/pXXX/pXXX_NNN.txt
    for f in vfiles:
        base = os.path.basename(f).split("_mic")[0]
        tp = os.path.join(PROJ, "data/vctk/txt", base.split("_")[0], base + ".txt")
        with open(tp, encoding="utf-8", errors="replace") as fh:
            vrefs.append(fh.readline().strip())
    from experiments.eval_asr import normalize
    vrefs = [normalize(r) for r in vrefs]
    OUT["vctk_original"] = eval_condition("vctk_ori", vwav, vspk, asv, refs=vrefs, asr=asr, wer_n=200)
    vprot = [protect(w, n=None, dist_range=(0.5, 0.6), rng=rng, mode="single")[0] for w in vwav]
    OUT["vctk_protected"] = eval_condition("vctk_pro", vprot, vspk, asv, refs=vrefs, asr=asr, wer_n=200)
    vatt = [attack_audio(p, rng) for p in vprot]
    OUT["vctk_attacked"] = eval_condition("vctk_att", vatt, vspk, asv)

    # ---------- 3. Fig.7 single 模式扫描（dev-clean 1000句）----------
    print("== 3. Fig.7 single 扫描（dev-clean 25人×40句）==", flush=True)
    dfiles, dspk, drefs, dwav = load_ls(os.path.join(
        PROJ, "data/librispeech/LibriSpeech/dev-clean/*/*"), n_spk=25, n_utt=40)
    fig7 = []
    for lo, hi in RANGES:
        pool = [gen_params(n=1, dist_range=(lo, hi), rng=rng) for _ in range(20)]
        if not pool:
            continue
        dp = [protect(w, rng=rng, param_pool=pool, mode="single")[0] for w in dwav]
        eer, _ = dataset_eer(asv, list(zip(dp, dspk)), max_imp_per_utt=1000)
        sel = np.random.default_rng(7).choice(len(dwav), 200, replace=False)
        wer, _, _ = corpus_wer(asr, [dp[i] for i in sel], [drefs[i] for i in sel])
        fig7.append({"lo": lo, "hi": hi, "eer": round(eer, 4), "wer": round(wer, 4)})
        print(f"  n=1 ({lo},{hi}): EER={eer*100:.2f}% WER={wer*100:.2f}%", flush=True)
    OUT["fig7_single"] = fig7

    # 图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mids = [(r["lo"] + r["hi"]) / 2 * 100 for r in fig7]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(mids, [r["wer"] * 100 for r in fig7], "o-", label="ASR WER (%)")
    ax.plot(mids, [r["eer"] * 200 for r in fig7], "s-", label="ASV error = 2×EER (%)")
    ax.set_xlabel("Distortion strength range (%)")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Fig.7 reproduction (single mode, n=1, dev-clean 1000 utts)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PROJ, "experiments/results/fig_exp1_fig7_single.png"), dpi=150)

    OUT["elapsed_s"] = round(time.time() - t_all, 1)
    with open(os.path.join(PROJ, "experiments/results/m3_complete.json"), "w",
              encoding="utf-8") as f:
        json.dump(OUT, f, ensure_ascii=False, indent=1)
    print("ALL DONE", OUT["elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    main()
