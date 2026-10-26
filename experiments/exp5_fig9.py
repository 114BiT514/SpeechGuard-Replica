"""[P1-2] 复现论文 Fig.9：EER × dist 区间 × n∈{2,4,6,8,10}（二维矩阵）。

论文 Fig.9 为"固定 n、扫 dist 十区间"的 EER 曲线族（0~0.5 轴），与 Fig.7（n=1
含 WER 双轴）互补，展示 n 对可 anonymity/quality 工作窗的影响。

规模按 M3 教训 §2：dev-clean 10 人 × 40 句=400 句（每人 ≥30 句、全对配对），
single 模式、constrained 池抽样（与主表同一实现）；EER=ECAPA。预计 ~60-80 分钟。

运行（GPU 空闲时）：
  E:/Miniconda/envs/speechguard/python.exe experiments/exp5_fig9.py
输出：experiments/results/exp5_fig9.json + fig_exp5_fig9.png
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.eval_asv import dataset_eer, load_asv  # noqa: E402
from src.acoustic.warping import gen_params, protect  # noqa: E402

RES = os.path.join(PROJ, "experiments", "results")
RANGES = [(round(0.1 * i, 1), round(0.1 * (i + 1), 1)) for i in range(10)]
NS = [2, 4, 6, 8, 10]
POOL = 40


def load_dev(n_spk=10, n_utt=40):
    import soundfile as sf
    wavs, spk = [], []
    for d in sorted(glob.glob(os.path.join(
            PROJ, "data", "librispeech", "LibriSpeech", "dev-clean", "*")))[:n_spk]:
        for f in sorted(glob.glob(os.path.join(d, "*", "*.flac")))[:n_utt]:
            wavs.append(np.asarray(sf.read(f)[0], np.float64))
            spk.append(os.path.basename(os.path.dirname(os.path.dirname(f))))
    return wavs, spk


def main():
    rng = np.random.default_rng(1949)
    t_all = time.time()
    wavs, spk = load_dev()
    print(f"dev-clean 子集：{len(wavs)} 句 / {len(set(spk))} 人", flush=True)
    asv = load_asv("cuda:0")

    e0, _ = dataset_eer(asv, list(zip(wavs, spk)))
    print(f"baseline EER={e0:.2%}", flush=True)

    # 参数池（先全部预建，高 dist 档拒绝慢，进度可见）
    pools = {}
    for n in NS:
        for lo, hi in RANGES:
            pool, tries = [], 0
            while len(pool) < POOL and tries < POOL * 400:
                tries += 1
                try:
                    pool.append(gen_params(n=n, dist_range=(lo, hi), rng=rng))
                except RuntimeError:
                    break
            pools[(n, lo, hi)] = pool
        print(f"  pools n={n} done ({time.time()-t_all:.0f}s)", flush=True)

    out = {"baseline_eer": round(e0, 4), "n_speakers": len(set(spk)),
           "utts_per_spk": 40, "curves": {}}
    for n in NS:
        curve = []
        for lo, hi in RANGES:
            t0 = time.time()
            prot = [protect(w, n=n, dist_range=(lo, hi), rng=rng,
                            param_pool=pools[(n, lo, hi)])[0] for w in wavs]
            eer, _ = dataset_eer(asv, list(zip(prot, spk)))
            curve.append({"dist": [lo, hi], "eer": round(eer, 4)})
            print(f"  n={n} ({lo},{hi}) EER={eer:.2%} ({time.time()-t0:.0f}s)",
                  flush=True)
        out["curves"][str(n)] = curve
        json.dump(out, open(os.path.join(RES, "exp5_fig9.json"), "w",
                            encoding="utf-8"), indent=2)   # 增量落盘防中断

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = [np.mean(c["dist"]) for c in out["curves"]["2"]]
    for n in NS:
        ys = [c["eer"] for c in out["curves"][str(n)]]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.4, label=f"n={n}")
    ax.axvspan(0.5, 0.6, color="#b45536", alpha=0.12, label="工作区 (0.5,0.6)")
    ax.set_xlabel("dist 区间中点")
    ax.set_ylabel("EER")
    ax.set_ylim(0, 0.55)
    ax.set_title("Fig.9 复现：EER 随 dist 区间与 n 的变化"
                 "（dev-clean 10人×40句，single 模式，ECAPA）", fontsize=10)
    ax.legend(fontsize=8, ncol=6, loc="lower right", frameon=False)
    ax.grid(alpha=0.25, ls="--")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_exp5_fig9.png"), dpi=150)
    print(f"done {time.time()-t_all:.0f}s -> exp5_fig9.json / fig_exp5_fig9.png")


if __name__ == "__main__":
    main()
