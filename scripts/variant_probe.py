"""M3 试验：不同扭曲变体的 EER/WER 权衡（寻找论文 3.58%WER+40%EER 的工作点）。

变体：
  A. 复谱扭曲, min_len=3, 仅有声段   （= 当前默认实现）
  B. 幅度扭曲+相位保持, min_len=3, 仅有声段
  C. 幅度扭曲+相位保持, min_len=1, 全段（含无声）
运行：E:/Miniconda/envs/speechguard/python.exe scripts/variant_probe.py
"""
import glob
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from experiments.eval_asr import corpus_wer, librispeech_ref, load_asr
from experiments.eval_asv import dataset_eer, load_asv
from src.acoustic.warping import _frame_view, _warp_segment, gen_params, multi_warp
from src.preprocess.vad import merge_short_segments, segmentation, voiced_flags

SR = 16000
FRAME_LEN = 576


def warp_seg_mag_only(seg, freq_map):
    """幅度谱扭曲（精细网格）+ 相位保持。"""
    n = seg.size
    nf = 4 * n
    fine = np.fft.rfft(seg, n=nf)
    omf = np.linspace(0, np.pi, nf // 2 + 1)
    tgt_fine = freq_map(omf)
    mag = np.interp(tgt_fine, omf, np.abs(fine))
    return np.fft.irfft(mag * np.exp(1j * np.angle(fine)), n=n)


def protect_variant(w, pool, rr, min_len, mag_only, voiced_only):
    frames, spec, _ = _frame_view(w)
    segs = segmentation(voiced_flags(spec))
    if min_len > 1:
        segs = merge_short_segments(segs, min_len)
    out = w.copy()
    for fs, fe, voiced in segs:
        if voiced_only and not voiced:
            continue
        s0, s1 = fs * FRAME_LEN, min((fe + 1) * FRAME_LEN, len(w))
        if s1 - s0 < 64:
            continue
        a, b, d = pool[rr.integers(len(pool))]
        fmap = lambda w_, a=a, b=b: multi_warp(w_, a, b)  # noqa: E731
        if mag_only:
            out[s0:s1] = warp_seg_mag_only(w[s0:s1], fmap)
        else:
            out[s0:s1] = _warp_segment(w[s0:s1], fmap)
    return out


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

    print("%-28s %10s %10s" % ("variant", "EER%", "WER%"), flush=True)
    for min_len in (1, 2, 3):
        for n in (1, 2, 3):
            pool = [gen_params(n=n, dist_range=(0.5, 0.6), rng=np.random.default_rng(1))
                    for _ in range(40)]
            rr = np.random.default_rng(11)
            prot = [protect_variant(w, pool, rr, min_len, False, True) for w in wavs]
            eer, _ = dataset_eer(asv, list(zip(prot, spk)))
            wer, _, _ = corpus_wer(asr, prot, refs)
            print("%-28s %10.2f %10.2f" % (f"复谱 min{min_len} 仅有声 n={n}",
                                           eer * 100, wer * 100), flush=True)


if __name__ == "__main__":
    main()
