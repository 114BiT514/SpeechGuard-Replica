"""[M2 任务5] warping 演示：真实语音 protect/recover + 频谱三联图。

运行：E:/Miniconda/envs/speechguard/python.exe scripts/warping_demo.py
产出：assets/warping_demo_{original,protected,recovered}.wav
      assets/fig_m2_warping_spectrogram.png（论文 Fig.6 风格）
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from src.acoustic.warping import protect, recover  # noqa: E402

SR = 16000
DUR = 2.5


def spec_db(x):
    from scipy.signal import stft as scipy_stft

    _, _, Z = scipy_stft(x, fs=SR, nperseg=512, noverlap=384)
    return 20 * np.log10(np.abs(Z) + 1e-10), _


def main():
    flac = sorted(glob.glob(os.path.join(PROJ, "data/librispeech/LibriSpeech/test-clean/*/*/*.flac")))[0]
    x, sr = sf.read(flac)
    assert sr == SR
    x = x[: int(DUR * SR)].astype(np.float64)

    prot, archive = protect(x, rng=np.random.default_rng(2026))
    rec = recover(prot, archive)
    snr = 10 * np.log10(np.sum(x ** 2) / max(np.sum((x - rec) ** 2), 1e-30))

    os.makedirs(os.path.join(PROJ, "assets"), exist_ok=True)
    sf.write(os.path.join(PROJ, "assets", "warping_demo_original.wav"), x, SR)
    sf.write(os.path.join(PROJ, "assets", "warping_demo_protected.wav"), prot, SR)
    sf.write(os.path.join(PROJ, "assets", "warping_demo_recovered.wav"), rec, SR)

    freqs, t = None, None
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, sharey=True)
    for ax, sig, title in zip(
        axes,
        [x, prot, rec],
        ["Original", f"Protected ({len(archive)} segments, dist∈[0.5,0.6])", f"Recovered (SNR {snr:.1f} dB)"],
    ):
        S, f = spec_db(sig)
        t = np.linspace(0, len(sig) / SR, S.shape[1])
        ax.imshow(S, aspect="auto", origin="lower", extent=[t[0], t[-1], f[0], f[-1]],
                  cmap="magma", vmin=-80, vmax=5, interpolation="nearest")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title, fontsize=10, loc="left")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    out = os.path.join(PROJ, "assets", "fig_m2_warping_spectrogram.png")
    fig.savefig(out, dpi=150)
    print("segments:", len(archive), "| n:", [s.n for s in archive])
    print("roundtrip SNR: %.1f dB" % snr)
    print("saved:", out)
    print("saved: assets/warping_demo_{original,protected,recovered}.wav")


if __name__ == "__main__":
    main()
