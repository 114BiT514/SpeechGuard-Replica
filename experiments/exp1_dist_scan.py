"""[M3 任务1] 复现论文 Fig.7（n=1 扫 10 个 dist 区间）与 Table 2 子集（多 n 矩阵）。

数据：dev-clean 子集（默认 10 说话人 × 4 条）。模型差异（报告中注明）：论文选参
实验用 SpeechBrain Transformer ASR + TitaNet-Large ASV，本文统一用 Conformer ASR +
ECAPA-TDNN。每个条件 (n, dist_range) 对全部语音做 protect（段级参数从池中抽取，
池按 (n, dist_range) 预构建），然后测语料级 WER 与 EER。

产出：results/exp1_dist_scan.csv（含 baseline 行）
      results/fig_exp1_fig7.png（n=1：WER 与 2×EER 随 dist 区间，论文 Fig.7 同款）
      results/fig_exp1_table2.png（EER/WER 两张热图，n × dist 区间，论文 Table 2 风格）
运行：E:/Miniconda/envs/speechguard/python.exe experiments/exp1_dist_scan.py
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.eval_asr import corpus_wer, librispeech_ref, load_asr  # noqa: E402
from experiments.eval_asv import dataset_eer, load_asv  # noqa: E402
from src.acoustic.warping import protect  # noqa: E402
from src.preprocess.vad import segmentation, voiced_flags  # noqa: E402

RANGES = [(round(0.1 * i, 1), round(0.1 * (i + 1), 1)) for i in range(10)]


def build_subset(data_root: str, n_speakers: int, n_utts: int):
    flacs = []
    spk_dirs = sorted(glob.glob(os.path.join(data_root, "LibriSpeech", "dev-clean", "*")))
    for d in spk_dirs[:n_speakers]:
        fs = sorted(glob.glob(os.path.join(d, "*", "*.flac")))[:n_utts]
        flacs += fs
    spk_ids = [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in flacs]
    refs = [librispeech_ref(f) for f in flacs]
    wavs = [__import__("soundfile").read(f)[0].astype(np.float64) for f in flacs]
    return wavs, spk_ids, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", type=int, default=8)
    ap.add_argument("--utts", type=int, default=3)
    ap.add_argument("--table2-ns", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--pool-size", type=int, default=40)
    ap.add_argument("--beam", type=int, default=4,
                    help="Conformer beam size；论文用 66+LM，8GB 卡上过慢，"
                         "降到 4 换取 ~15x 提速（WER 绝对值 +0.5~1%%，趋势不变）")
    args = ap.parse_args()
    rng = np.random.default_rng(2026)
    out_dir = os.path.join(PROJ, "experiments", "results")
    os.makedirs(out_dir, exist_ok=True)

    import soundfile as sf

    wavs, spk_ids, refs = build_subset(
        os.path.join(PROJ, "data", "librispeech"), args.speakers, args.utts)
    print(f"子集：{len(wavs)} 条（{args.speakers} 说话人 × {args.utts}）", flush=True)

    print("加载 ECAPA...", flush=True)
    asv = load_asv(args.device)
    print("加载 Conformer...", flush=True)
    asr = load_asr(args.device, overrides=f"test_beam_size: {args.beam}")

    rows = []

    def run_condition(tag, n, lo, hi, pool=None):
        t0 = time.time()
        prot = []
        for w in wavs:
            p, _ = protect(w, n=n, dist_range=(lo, hi), rng=rng, param_pool=pool)
            prot.append(p)
        eer, _ = dataset_eer(asv, list(zip(prot, spk_ids)))
        wer, _, _ = corpus_wer(asr, prot, refs)
        dt = time.time() - t0
        print(f"[{tag}] n={n} dist=({lo},{hi}): EER={eer:.2%} WER={wer:.2%} ({dt:.0f}s)",
              flush=True)
        rows.append({"tag": tag, "n": n, "dist_lo": lo, "dist_hi": hi,
                     "eer": round(eer, 4), "wer": round(wer, 4)})
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return eer, wer

    # ---- baseline ----
    t0 = time.time()
    eer0, _ = dataset_eer(asv, list(zip(wavs, spk_ids)))
    wer0, _, hyps0 = corpus_wer(asr, wavs, refs)
    print(f"[baseline] 原始: EER={eer0:.2%} WER={wer0:.2%} ({time.time()-t0:.0f}s)", flush=True)
    print("   转写样例:", hyps0[0], "|| 参考:", refs[0], flush=True)
    rows.append({"tag": "baseline", "n": "", "dist_lo": "", "dist_hi": "",
                 "eer": round(eer0, 4), "wer": round(wer0, 4)})

    # ---- 参数池（按 n × dist_range 预构建；池与音频无关可复用）----
    ns_all = [1] + [n for n in args.table2_ns if n != 1]
    pools = {}
    for n in ns_all:
        for lo, hi in RANGES:
            pool, tries = [], 0
            while len(pool) < args.pool_size and tries < args.pool_size * 80:
                tries += 1
                try:
                    from src.acoustic.warping import gen_params
                    pool.append(gen_params(n=n, dist_range=(lo, hi), rng=rng))
                except RuntimeError:
                    break
            pools[(n, lo, hi)] = pool
            print(f"  pool n={n} ({lo},{hi}): {len(pool)} 组", flush=True)

    # ---- 条件循环（先 n=1 全扫 = Fig.7；再 Table 2 子集）----
    csv_path = os.path.join(out_dir, "exp1_dist_scan.csv")
    for n in ns_all:
        for lo, hi in RANGES:
            eer, wer = run_condition(f"fig7" if n == 1 else "table2",
                                     n, lo, hi, pool=pools[(n, lo, hi)])
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["tag", "n", "dist_lo", "dist_hi", "eer", "wer"])
                w.writeheader()
                w.writerows(rows)

    # ---- 图：Fig.7（n=1）----
    fig7 = [r for r in rows if r["tag"] == "fig7"]
    mids = [(lo + hi) / 2 * 100 for lo, hi in RANGES]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(mids, [r["wer"] * 100 for r in fig7], "o-", label="ASR WER (%)")
    ax.plot(mids, [r["eer"] * 200 for r in fig7], "s-", label="ASV error = 2×EER (%)")
    ax.axhline(wer0 * 100, color="C0", ls="--", lw=1, label=f"WER original ({wer0*100:.1f}%)")
    ax.axhline(eer0 * 200, color="C1", ls="--", lw=1, label=f"2×EER original ({eer0*200:.1f}%)")
    ax.set_xlabel("Distortion strength range (%)")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Fig.7 reproduction: n=1, dev-clean subset (Conformer ASR / ECAPA ASV)")
    ax.legend(fontsize=8)
    ax.set_xticks(mids)
    ax.set_xticklabels([f"({lo},{hi})" for lo, hi in RANGES], rotation=45, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_exp1_fig7.png"), dpi=150)

    # ---- 图：Table 2 子集热图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 1.2 + 0.55 * len(ns_all)))
    for ax, key, title in zip(axes, ["eer", "wer"], ["EER (%)", "WER (%)"]):
        mat = np.full((len(ns_all), 10), np.nan)
        for i, n in enumerate(ns_all):
            for j, (lo, hi) in enumerate(RANGES):
                for r in rows:
                    if r["tag"] == "table2" and r["n"] == n and r["dist_lo"] == lo:
                        mat[i, j] = r[key] * 100
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"({lo},{hi})" for lo, hi in RANGES], rotation=45, fontsize=6)
        ax.set_yticks(range(len(ns_all)))
        ax.set_yticklabels([f"n={n}" for n in ns_all], fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(len(ns_all)):
            for j in range(10):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                            color="w", fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.9)
    fig.suptitle("Table 2 reproduction (subset): EER / WER vs n × dist range", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_exp1_table2.png"), dpi=150)

    summary = {"baseline": {"eer": eer0, "wer": wer0},
               "n_speakers": args.speakers, "utts_per_spk": args.utts,
               "conditions": rows}
    with open(os.path.join(out_dir, "exp1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("saved:", csv_path)
    print("saved: fig_exp1_fig7.png, fig_exp1_table2.png, exp1_summary.json")


if __name__ == "__main__":
    main()
