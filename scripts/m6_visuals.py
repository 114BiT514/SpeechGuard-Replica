"""[M6 任务2] 报告可视化素材一键生成（真实语音，走完整管线）。

产出（assets/）：
  fig_m6_spectrogram.png     原始/保护/攻击/恢复 四联频谱（Fig.6 风格）
  fig_m6_timeline.png        敏感词时间轴（发布页素材）
  fig_m6_warp.png            warping 映射曲线（原始/保护/逆扭曲对照）
  pipeline_demo/             端到端产物目录（summary.json + 三态 mp3/wav）

运行（GPU）：E:/Miniconda/envs/speechguard/python.exe scripts/m6_visuals.py
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from src import pipeline  # noqa: E402
from src.acoustic.warping import PI, distortion_strength, multi_warp  # noqa: E402
from src.viz import sensitive_timeline, spectrogram_row  # noqa: E402

ASSETS = os.path.join(PROJ, "assets")


def pick_utterance():
    """选一条含 ≥2 敏感项的真实语音（test-clean）。"""
    from src.content.detect import detect_sensitive
    from src.content.transcribe import transcribe_with_times
    files = sorted(glob.glob(os.path.join(
        PROJ, "data/librispeech/LibriSpeech/test-clean/*/*/*.flac")))
    for f in files[:600]:
        wav, _ = sf.read(f)
        wav = np.asarray(wav, np.float32)
        ctc = transcribe_with_times(wav, device="cuda:0")
        hits = detect_sensitive(ctc, device="cuda:0")
        if len(hits) >= 2 and wav.size / 16000 <= 12:
            return f, wav, hits
    raise RuntimeError("no suitable utterance")


def main():
    f, wav, hits = pick_utterance()
    print("utt:", os.path.basename(f), f"{wav.size/16000:.1f}s",
          "hits:", [h.word for h in hits], flush=True)

    outdir = os.path.join(ASSETS, "pipeline_demo")
    summary = pipeline.run_pipeline(f, outdir, device="cuda:0")

    bundle = pipeline.protect(wav, device="cuda:0")
    prot = pipeline._to_float(bundle["wav"])
    from src.acoustic.warping import SegmentParams, _apply_single, gen_params, recover
    rng = np.random.default_rng(2)
    a, b, d = gen_params(rng=rng)
    fake = [SegmentParams(0, max(wav.size // 576, 1), a, b, d)]
    att, _ = _apply_single(prot.astype(np.float64), archive=fake, invert=True)
    rec = recover(prot.astype(np.float64), bundle["archive"])

    spectrogram_row([(wav.astype(np.float64), "原始"),
                     (prot.astype(np.float64), "保护后"),
                     (att.astype(np.float64), "reducing 攻击后"),
                     (rec, "按密钥恢复")],
                    os.path.join(ASSETS, "fig_m6_spectrogram.png"))
    print("fig_m6_spectrogram.png")

    sensitive_timeline(bundle["ann"], wav.size / 16000,
                       os.path.join(ASSETS, "fig_m6_timeline.png"))
    print("fig_m6_timeline.png")

    grid = np.linspace(0.0, PI, 2001)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), dpi=150)
    ax = axes[0]
    for i in range(min(4, len(bundle["archive"]))):
        seg = bundle["archive"][i]
        ax.plot(grid, multi_warp(grid, seg.alpha, seg.beta), lw=1.2,
                label=f"段{i} (n={seg.n}, dist={seg.dist:.3f})")
    ax.plot(grid, grid, "k--", lw=1, label="恒等")
    ax.set_xlabel("ω (rad)"); ax.set_ylabel("m(ω)")
    ax.set_title("多参数分段线性扭曲 m(ω,α,β)（保护）", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax = axes[1]
    seg = bundle["archive"][0]
    ax.plot(grid, multi_warp(grid, seg.beta, seg.alpha), lw=1.5, color="#b45536",
            label="逆映射 m⁻¹(ω,α,β)=m(ω,β,α)")
    ax.plot(grid, grid, "k--", lw=1, label="恒等")
    ax.set_xlabel("ω (rad)"); ax.set_ylabel("m⁻¹(ω)")
    ax.set_title("逆扭曲（α/β 互换即恢复密钥）", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_m6_warp.png"))
    plt.close(fig)
    print("fig_m6_warp.png")
    print("demo summary:", summary.get("sensitive_items"), summary.get("rtc"))


if __name__ == "__main__":
    main()
