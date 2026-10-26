"""[P2-1] VoiceMask 基线评估：Fig.10 攻击成功率 + Fig.11 风格 EER/WER + RTC。

攻击成功率（纯 CPU，10,000 trial）：判据与 exp4 同语义"攻击后离原始更近=成功"，
失真用**信号域归一化 MSE 代理**（时域方法没有频域 dist(m⁻¹∘m) 的定义——
论文未公开其对 VoiceMask 的具体判据，此为近似口径，报告注明）：
  dist_protect = d(masked, original)；dist_attack(L') = d(recover(masked, L'), original)
  成功 = dist_attack < dist_protect（猜对 L 或整数倍/约数对齐都会触发，
  理论下界 ≈ P(L'=L)+部分对齐 ≈ 2-4%）。

EER/WER（GPU，exp4 同协议：test-clean 12人×12句、single 映射框架不变、
ECAPA 全对+1000 imposter、Conformer beam=4、WER 30 句子集）：
  prot  = masked；att = 随机错误 L' 的伪恢复；rec = 真参数恢复。
结果 merge 进 exp4_attack_rates.json["rates"]["voicemask"] 与
exp4_compare.json["voicemask"]，并重绘 fig_exp4_fig10.png（VoiceMask† 注脚）。

运行：
  attack:  E:/Miniconda/envs/speechguard/python.exe experiments/exp6_voicemask.py --attack
  全量:    E:/Miniconda/envs/speechguard/python.exe experiments/exp6_voicemask.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from src.baselines import voicemask as VM  # noqa: E402

RES = os.path.join(PROJ, "experiments", "results")
N_TRIALS = 10000


def attack_rate(rng, n_trials=N_TRIALS, utts=3):
    """3 条真实语音（~6s）各 10000 trial 的均值。"""
    files = sorted(glob.glob(os.path.join(
        PROJ, "data", "librispeech", "LibriSpeech", "test-clean", "1089", "*", "*.flac")))[:utts]
    rates = []
    for f in files:
        w = np.asarray(sf.read(f)[0], np.float64)
        prot, params = VM.protect(w, rng=rng)
        d_prot = VM.distortion_to_original(prot, w)
        wins = 0
        for _ in range(n_trials):
            g = VM.attack_guess(rng)
            d_att = VM.distortion_to_original(VM.recover(prot, g), w)
            wins += d_att < d_prot
        rates.append(wins / n_trials)
    return float(np.mean(rates)), rates


def main(with_eer=True):
    rng = np.random.default_rng(2026)
    out = {}
    t0 = time.time()
    rate, per_utt = attack_rate(rng)
    print(f"VoiceMask reducing-attack success = {rate*100:.2f}% "
          f"(per-utt {['%.2f%%' % (r*100) for r in per_utt]}, {time.time()-t0:.0f}s)",
          flush=True)
    ar_path = os.path.join(RES, "exp4_attack_rates.json")
    ar = json.load(open(ar_path, encoding="utf-8"))
    ar["rates"]["voicemask"] = {
        "success_rate": round(rate, 4), "per_utt": [round(r, 4) for r in per_utt],
        "trials": N_TRIALS,
        "note": "†时域方法：失真判据为信号域 MSE 代理（非频域 dist 定义），"
                "绝对值不与其余四家严格可比，只作空间自由度排序参考"}
    json.dump(ar, open(ar_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    out["attack"] = ar["rates"]["voicemask"]

    if with_eer:
        from experiments.eval_asr import corpus_wer, librispeech_ref, load_asr, normalize
        from experiments.eval_asv import dataset_eer, load_asv
        dirs = sorted(glob.glob(os.path.join(
            PROJ, "data", "librispeech", "LibriSpeech", "test-clean", "*", "*")))
        picked = []
        for d in dirs[:12]:
            picked += sorted(glob.glob(os.path.join(d, "*.flac")))[:12]
        wavs, spk, refs = [], [], []
        for f in picked:
            wavs.append(np.asarray(sf.read(f)[0], np.float64))
            spk.append(os.path.basename(os.path.dirname(f)))
            refs.append(librispeech_ref(f))
        print(f"EER/WER 协议：{len(wavs)} 句 / {len(set(spk))} 人", flush=True)
        asv = load_asv("cuda:0")
        asr = load_asr(device="cuda:0", overrides="test_beam_size: 4")

        prots, atts, recs = [], [], []
        tp = tr = 0.0
        rng2 = np.random.default_rng(7)
        for w in wavs:
            t0 = time.time()
            p, params = VM.protect(w, rng=rng2)
            tp += time.time() - t0
            prots.append(p)
            g = VM.attack_guess(rng2)
            while g == params:
                g = VM.attack_guess(rng2)
            atts.append(VM.recover(p, g))
            t1 = time.time()
            recs.append(VM.recover(p, params))
            tr += time.time() - t1

        def ev(tag, xs, do_wer=True):
            eer, _ = dataset_eer(asv, list(zip(xs, spk)), max_imp_per_utt=1000)
            rec = {"eer": round(eer, 4)}
            if do_wer:
                idx = np.random.default_rng(7).choice(len(xs), min(30, len(xs)), replace=False)
                wer, _, _ = corpus_wer(asr, [xs[i] for i in idx], [refs[i] for i in idx])
                rec["wer"] = round(wer, 4)
            print(f"  [{tag}] EER={eer*100:.2f}%"
                  + (f" WER={rec['wer']*100:.2f}%" if "wer" in rec else ""), flush=True)
            return rec

        cmp_path = os.path.join(RES, "exp4_compare.json")
        cmpd = json.load(open(cmp_path, encoding="utf-8"))
        cmpd["voicemask"] = {
            "prot": ev("vm/prot", prots), "att": ev("vm/att", atts),
            "rec": ev("vm/rec", recs),
            "rtc": {"protect": round(tp / len(wavs), 4),
                    "recover": round(tr / len(wavs), 4)},
            "note": "exp4 同协议（12人×12句子集，相对比较）"}
        json.dump(cmpd, open(cmp_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        out["compare"] = cmpd["voicemask"]
        _redraw_fig10(ar, cmpd)


def _redraw_fig10(ar, cmpd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    order = [("single", "单参数\n(公式2)"), ("mcadams", "McAdams"),
             ("overo", "Overo"), ("voicemask", "VoiceMask†"),
             ("speechguard_raw", "SpeechGuard\n(raw 抽样)"),
             ("speechguard_constrained", "SpeechGuard\n(本实现)")]
    vals = [ar["rates"][k]["success_rate"] * 100 for k, _ in order if k in ar["rates"]]
    labels = [lab for k, lab in order if k in ar["rates"]]
    colors = ["#384f5b"] * len(vals)
    sg_i = [i for i, (k, _) in enumerate(order) if k.startswith("speechguard_raw")]
    if sg_i:
        colors[sg_i[0]] = "#b45536"
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}",
                ha="center", fontsize=9)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Attack success rate (%)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("Fig.10 复现（含 VoiceMask）：10,000 次 reducing 攻击成功率\n"
                 "†VoiceMask 为时域方法，失真判据用信号 MSE 代理（口径见 json note）",
                 fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_exp4_fig10.png"), dpi=150)
    plt.close(fig)
    print("fig_exp4_fig10.png redrawn")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", action="store_true", help="仅跑 CPU 攻击成功率")
    args = ap.parse_args()
    main(with_eer=not args.attack)
