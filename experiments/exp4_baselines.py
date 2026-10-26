"""[M5 任务5] 基线对比（论文 Fig.10/11 + Table 6）。

A 部分（CPU）：10,000 次 reducing 攻击成功率——纯映射复合仿真，判据与论文一致
  dist(m_att⁻¹ ∘ m_true) < dist(m_true)。方法×参数空间：
    single   公式(2)，α∈[0.6,1.7]
    mcadams  正弦频谱移位 warp（单调约束 A·2π/P<1）
    overo    双参数分段线性（α,β 独立均匀）
    speechguard  多参数 n∈[1,10] dist∈(0.5,0.6)：raw（论文原味抽样）
                 与 constrained（本实现斜率约束）并报——exp2 已量化该差异。
  论文 Fig.10：SpeechGuard 最低，比次好(Overo)还低 6.80pp。

B 部分（GPU，与 M3 全集协议不同——规模 12人×12句，只看方法间相对结论）：
  各方法 protect → attack(随机一次逆扭曲) → recover，ECAPA EER（三态）
  + Conformer WER（prot/att 两态，30 句子集）+ 单句 RTC。
  论文 Fig.11（三数据集平均）：Prot EER SG 33.48≈VM 33.42 > Overo 22.97 >
  McAdams 16.91；SG 攻击后 EER 一致最高(35.88)，其余方法至少一个数据集反而下降。
  注意：小子集会系统性低估 EER（M3 教训 §2），本表用于方法间**相对比较**。

运行：
  E:/Miniconda/envs/speechguard/python.exe experiments/exp4_baselines.py [--gpu N]
  （--gpu N：仅 B 部分，N=人数×句数；默认全部）
输出：results/exp4_attack_rates.json / fig_exp4_fig10.png / exp4_compare.json
"""
from __future__ import annotations

import argparse
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

from src.acoustic.warping import (DIST_RANGE, PI, distortion_strength,  # noqa: E402
                                  gen_params)
from src.baselines import methods as B  # noqa: E402

RES = os.path.join(PROJ, "experiments", "results")
GRID = np.linspace(0.0, PI, 4001)
N_TRIALS = 10000


# ───────────────────────── A：攻击成功率 ─────────────────────────

def _d_of(fwd_tab):
    return float(np.trapz(np.abs(fwd_tab - GRID), GRID))


def _interp_inv(fwd_tab):
    grid = GRID
    def finv(y):
        return np.interp(y, fwd_tab, grid)
    return finv


def sample_map(method, rng, sampling="constrained", max_trials=200000):
    """dist 校准抽样（论文协议：保护与攻击者先验都限定 dist∈(0.5,0.6)）。

    返回 (fwd_tab=f(GRID), inv_fn, dist)。方法空间自由度不同：
    single 1 维 → 约束后剩 ~0 维；overo 2 维 → 1 维；mcadams 3 维 → 2 维；
    speechguard 2n 维 n∈[1,10] → 平均 ~10 维。成功率差异正是要度量的东西。
    """
    lo, hi = DIST_RANGE
    from src.acoustic.warping import multi_warp
    for _ in range(max_trials):
        if method == "single":
            alpha = float(rng.uniform(0.5, 2.0))
            from src.acoustic.warping import piecewise_warp
            fwd = piecewise_warp(GRID, alpha)
        elif method == "mcadams":
            A = rng.uniform(0.02, 0.4) * PI
            P = rng.uniform(0.25, 1.6) * PI
            c = rng.uniform(0.05, 0.95) * PI
            if A * 2 * PI / P >= 0.98:
                continue
            fwd = GRID + A * np.sin(2 * PI * (GRID - c) / P)
            fwd = fwd - fwd[0]
            fwd = fwd * (PI / fwd[-1])            # 重标定端点到 [0,π]
        elif method == "overo":
            a = float(rng.uniform(1e-3, PI - 1e-3))
            b = float(rng.uniform(1e-3, PI - 1e-3))
            fwd = multi_warp(GRID, np.array([a]), np.array([b]))
        elif method == "speechguard":
            if sampling == "raw":
                n = int(rng.integers(1, 11))
                a = np.sort(rng.uniform(1e-3, PI - 1e-3, n))
                b = np.sort(rng.uniform(1e-3, PI - 1e-3, n))
                fwd = multi_warp(GRID, a, b)
                alpha, beta = a, b
            else:
                alpha, beta, d = gen_params(rng=rng)  # 内部已拒绝到 DIST_RANGE
                fwd = multi_warp(GRID, alpha, beta)
                return fwd, lambda y, a2=alpha, b2=beta: multi_warp(y, b2, a2), d
        else:
            raise KeyError(method)
        d = _d_of(fwd)
        if lo <= d <= hi:
            if method == "speechguard":
                return fwd, lambda y, a2=alpha, b2=beta: multi_warp(y, b2, a2), d
            return fwd, _interp_inv(fwd), d
    raise RuntimeError(f"sample_map({method}) rejection 抽样失败")


def attack_rate(method, rng, sampling="constrained", n_trials=N_TRIALS):
    """成功率 = #{dist(m_att⁻¹∘m_true) < dist(m_true)}，双方参数独立同先验。"""
    fwd_true, _, d_true = sample_map(method, rng, sampling)
    wins = 0
    dists = np.empty(n_trials)
    for i in range(n_trials):
        _, inv_att, _ = sample_map(method, rng, sampling)
        comp = inv_att(fwd_true)
        dists[i] = _d_of(comp)
        wins += dists[i] < d_true
    return wins / n_trials, dists, d_true


def part_a():
    rng = np.random.default_rng(2026)
    out = {"trials": N_TRIALS, "protocol":
           "reducing 攻击（论文 §5.2 / Fig.10）：判据 dist(m_att⁻¹∘m_true)<dist(m_true)；"
           "保护与攻击先验均校准 dist∈(0.5,0.6)；speechguard 双抽样并报"
           "（raw=论文独立排序均匀抽样；constrained=本实现斜率约束，代价见 exp2/M3 findings）",
           "rates": {}}
    rows = []
    specs = [("single", "constrained"), ("mcadams", "constrained"),
             ("overo", "constrained"), ("speechguard", "raw"),
             ("speechguard", "constrained")]
    for m, s in specs:
        t0 = time.time()
        rate, dists, dtrue = attack_rate(m, rng, s)
        key = m if s != "constrained" or m == "speechguard" else m
        if m == "speechguard":
            key = f"speechguard_{s}"
        out["rates"][key] = {"success_rate": round(rate, 4),
                             "median_dist_after_attack": float(np.median(dists)),
                             "dist_true": round(dtrue, 4),
                             "seconds": round(time.time() - t0, 1)}
        rows.append((key, rate))
        print(f"{key:22s} attack success = {rate*100:5.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(os.path.join(RES, "exp4_attack_rates.json"), "w",
                        encoding="utf-8"), indent=2)

    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = ["单参数\n(公式2)", "McAdams", "Overo",
              "SpeechGuard\n(raw 抽样)", "SpeechGuard\n(本实现)"]
    vals = [r * 100 for _, r in rows]
    bars = ax.bar(range(len(rows)), vals, color="#384f5b", width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}",
                ha="center", fontsize=9)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Attack success rate (%)")
    ax.set_ylim(0, max(65, max(vals) * 1.15))
    ax.set_title("Fig.10 复现：10,000 次 reducing 攻击成功率"
                 "（论文：SG 最低，比 Overo 低 6.8pp）", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_exp4_fig10.png"), dpi=150)
    print("saved fig_exp4_fig10.png / exp4_attack_rates.json")


# ───────────────────────── B：EER/WER 对比（GPU） ─────────────────────────

def part_b(n_spk=12, n_utt=12, wer_n=30):
    import soundfile as sf
    from experiments.eval_asr import corpus_wer, librispeech_ref, load_asr
    from experiments.eval_asv import load_asv, dataset_eer
    from src.acoustic.warping import protect as sg_protect, recover as sg_recover

    files = sorted(glob.glob(os.path.join(
        PROJ, "data/librispeech/LibriSpeech/test-clean", "*", "*")))
    picked = []
    for d in files[:n_spk]:
        us = sorted(glob.glob(os.path.join(d, "*.flac")))[:n_utt]
        picked += us
    print(f"B 部分：{len(picked)} 句 / 最多 {n_spk} 说话人", flush=True)

    data, spk = [], []
    for f in picked:
        w, _ = sf.read(f)
        data.append(np.asarray(w, np.float64))
        spk.append(int(os.path.basename(os.path.dirname(f))))

    asv = load_asv("cuda:0")
    asr = load_asr(device="cuda:0", overrides="test_beam_size: 4")
    rng = np.random.default_rng(2026)

    def cond_eval(tag, wavs, do_wer=True):
        eer, _ = dataset_eer(asv, list(zip(wavs, spk)), max_imp_per_utt=1000)
        rec = {"eer": round(eer, 4)}
        if do_wer:
            idx = rng.choice(len(wavs), min(wer_n, len(wavs)), replace=False)
            refs = [librispeech_ref(picked[i]) for i in idx]
            wer, _, _ = corpus_wer(asr, [wavs[i] for i in idx], refs)
            rec["wer"] = round(wer, 4)
        print(f"  [{tag}] EER={eer*100:.2f}%"
              + (f" WER={rec.get('wer',0)*100:.2f}%" if "wer" in rec else ""),
              flush=True)
        return rec

    out = {}
    from src.acoustic.warping import SegmentParams, _apply_single
    for method in ("single", "mcadams", "overo", "speechguard"):
        prots, atts, recs = [], [], []
        tp = tr = 0.0
        for w in data:
            if method == "speechguard":
                t0 = time.time()
                p, arch = sg_protect(w, rng=rng)
                tp += time.time() - t0
                a, b, d = gen_params(rng=rng)
                fake = [SegmentParams(0, max(w.size // 576, 1), a, b, d)]
                ap, _ = _apply_single(p, archive=fake, invert=True)
                atts.append(ap)
                t1 = time.time()
                recs.append(sg_recover(p, arch))
                tr += time.time() - t1
            else:
                fwd_tab, finv, d = sample_map(method, rng)
                _, ginv, _ = sample_map(method, rng)
                fwd_fn = lambda w, t=fwd_tab: np.interp(w, GRID, t)
                t0 = time.time()
                p = _apply_audio_map(w, fwd_fn)
                tp += time.time() - t0
                atts.append(_apply_audio_map(p, ginv))
                t1 = time.time()
                recs.append(_apply_audio_map(p, finv))
                tr += time.time() - t1
            prots.append(p)
        entry = {}
        entry["prot"] = cond_eval(f"{method}/prot", prots)
        entry["att"] = cond_eval(f"{method}/att", atts)
        entry["rec"] = cond_eval(f"{method}/rec", recs)
        n = len(data)
        entry["rtc"] = {"protect": round(tp / n, 4), "recover": round(tr / n, 4)}
        out[method] = entry
    json.dump(out, open(os.path.join(RES, "exp4_compare.json"), "w",
                        encoding="utf-8"), indent=2)
    print("saved exp4_compare.json")


def _apply_audio_map(wav, map_fn):
    """在音频上施加频率映射（整句一段映射，基线统一框架；SG 用 _apply_single）。"""
    from src.baselines.methods import _apply_maps
    wav = np.asarray(wav, dtype=np.float64).ravel()
    n_fr = int(np.ceil(wav.size / 576))
    ident = lambda w: w  # noqa: E731
    return _apply_maps(wav, [(0, n_fr, map_fn, ident)], invert=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", nargs="?", const=1, type=int,
                    help="仅跑 B 部分（人数）")
    args = ap.parse_args()
    if args.gpu is None:
        part_a()
        part_b()
    else:
        part_b(n_spk=args.gpu)
