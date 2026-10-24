"""M3 试验 2：OLA（加窗重叠相加）方案的保护端 EER/WER。

背景：段级矩形块方案的 roundtrip 较好，但 9 格扫描显示其 EER-WER 工作点远离论文
（EER≥9% 时 WER≥73%）。OLA 方案的合成平滑（WER 应低）、按段/按帧的时变扭曲
（EER 应升），其恢复端有损——与论文自述的恢复 WER 9.9% 一致，因此以保护端
指标为主要验收。

变体：
  D. OLA + 按段参数（min3 仅有声）
  E. OLA + 每帧参数（288-hop 帧各自独立）
运行：E:/Miniconda/envs/speechguard/python.exe scripts/variant_probe2.py
"""
import glob
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from experiments.eval_asr import corpus_wer, librispeech_ref, load_asr
from experiments.eval_asv import dataset_eer, load_asv
from src.acoustic.warping import gen_params, multi_warp
from src.preprocess.stft import frames_centered, istft, sqrt_hann
from src.preprocess.vad import merge_short_segments, segmentation, voiced_flags

FRAME_LEN = 576
HOP = 288


def ola_protect(w, pool, rr, min_len, per_frame):
    """OLA 方案保护：帧谱（加窗）按段/按帧扭曲后 istft 合成。"""
    frames, pad = frames_centered(w)                     # hop=288
    spec = np.fft.rfft(frames * sqrt_hann(FRAME_LEN), axis=-1)
    omega = np.linspace(0, np.pi, spec.shape[1])
    omf = np.linspace(0, np.pi, (4 * FRAME_LEN) // 2 + 1)
    new_spec = spec.copy()
    if per_frame:
        for i in range(spec.shape[0]):
            a, b, _ = pool[rr.integers(len(pool))]
            fine = np.fft.rfft(frames[i] * sqrt_hann(FRAME_LEN), n=4 * FRAME_LEN)
            tgt = multi_warp(omega, a, b)
            new_spec[i] = np.interp(tgt, omf, fine.real) + 1j * np.interp(tgt, omf, fine.imag)
    else:
        flags = voiced_flags(spec)
        segs = merge_short_segments(segmentation(flags), min_len)
        for fs, fe, voiced in segs:
            if not voiced:
                continue
            i0, i1 = fs * 2, (fe + 1) * 2                # 576-hop → 288-hop 帧号
            i1 = min(i1, spec.shape[0])
            if i1 - i0 < 2:
                continue
            a, b, _ = pool[rr.integers(len(pool))]
            for i in range(i0, i1):
                fine = np.fft.rfft(frames[i] * sqrt_hann(FRAME_LEN), n=4 * FRAME_LEN)
                tgt = multi_warp(omega, a, b)
                new_spec[i] = (np.interp(tgt, omf, fine.real)
                               + 1j * np.interp(tgt, omf, fine.imag))
    return istft(new_spec, target_len=w.size)


def main():
    flacs = []
    spk_dirs = sorted(glob.glob(os.path.join(PROJ, "data/librispeech/LibriSpeech/dev-clean/*")))[:8]
    for d in spk_dirs:
        flacs += sorted(glob.glob(os.path.join(d, "*", "*.flac")))[:3]
    spk = [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in flacs]
    refs = [librispeech_ref(f) for f in flacs]
    import soundfile as sf

    wavs = [sf.read(f)[0].astype(np.float64) for f in flacs]
    pool = [gen_params(n=1, dist_range=(0.5, 0.6), rng=np.random.default_rng(1))
            for _ in range(40)]

    asv = load_asv("cuda:0")
    asr = load_asr("cuda:0", overrides="test_beam_size: 4")

    print("%-30s %10s %10s" % ("variant", "EER%", "WER%"), flush=True)
    for tag, min_len, per_frame in (
        ("D OLA 按段参数 min3 仅有声", 3, False),
        ("E OLA 每帧参数", 0, True),
    ):
        rr = np.random.default_rng(11)
        prot = [ola_protect(w, pool, rr, min_len, per_frame) for w in wavs]
        eer, _ = dataset_eer(asv, list(zip(prot, spk)))
        wer, _, _ = corpus_wer(asr, prot, refs)
        print("%-30s %10.2f %10.2f" % (tag, eer * 100, wer * 100), flush=True)


if __name__ == "__main__":
    main()
