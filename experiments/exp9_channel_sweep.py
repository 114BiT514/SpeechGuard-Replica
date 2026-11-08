"""[M7 加分项 ②] 控制变量扫描：信道条件 vs 恢复质量曲线族。

固定其余条件，扫三个维度（执行计划 M7 加分项第 2 层）：
  码率   mp3 {32, 64, 128, 192} kbps
  信噪比 加性白噪 SNR {0, 5, 10, 20} dB
  带宽   重采样往返 {8, 11.025, 22.05} kHz
每个扫描点测四条曲线：
  recovered-WER（Conformer beam=4，论文口径）      —— 恢复质量
  recovered-EER（ECAPA，recovered 音频 vs 原说话人标签）—— 声纹找回程度
  sim(orig)（ECAPA 余弦均值）                      —— 逐句版"恢复保真"
  A2-FNR（加密→信道→按原帧号解密后敏感词隐藏率）    —— 内容隐私是否仍成立
样本与 exp8 同源同种子（protect 参数流一致，数字可互相印证）。

运行（GPU，约 30-40 分钟，转写磁盘缓存断点续跑）：
  E:/Miniconda/envs/speechguard/python.exe experiments/exp9_channel_sweep.py
输出：experiments/results/exp9_sweep.json + fig_exp9_curves.png
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.eval_asr import librispeech_ref, normalize  # noqa: E402
from experiments.exp8_channel_cases import (  # noqa: E402
    HYP_CACHE, SEED, SR, _add_noise, _mp3_roundtrip, _resample,
    pick_a1_files, pick_a2_files, transcribe_cached)

RES = os.path.join(PROJ, "experiments", "results")

# 维度定义：name -> [(label, fn)]
def _mk_mp3(k):
    return lambda w: _mp3_roundtrip(w, k)


def _mk_noise(s):
    rng = np.random.default_rng(SEED)
    return lambda w: _add_noise(w, s, rng)


def _mk_rs(r):
    return lambda w: _resample(_resample(w, SR, r), r, SR)


SWEEPS = {
    "bitrate_kbps": [("32", _mk_mp3(32)), ("64", _mk_mp3(64)),
                     ("128", _mk_mp3(128)), ("192", _mk_mp3(192))],
    "noise_snr_db": [("0", _mk_noise(0)), ("5", _mk_noise(5)),
                     ("10", _mk_noise(10)), ("20", _mk_noise(20))],
    "bandwidth_khz": [("8.0", _mk_rs(8000)), ("11.025", _mk_rs(11025)),
                      ("22.05", _mk_rs(22050))],
}


def main():
    import jiwer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.eval_asv import dataset_eer, load_asv
    from experiments.eval_asr import load_asr
    from src.acoustic.warping import protect, recover
    from src.content.annotation import Annotation
    from src.content.encrypt import decrypt_spans, encrypt_spans, gen_key
    from src.content.sensitive_lists import SensitiveLists

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    rng = np.random.default_rng(SEED)

    # ── 样本（与 exp8 同流）──
    a1 = []
    for f, spk in pick_a1_files():
        wav, _ = sf.read(f)
        prot, arch = protect(np.asarray(wav, np.float64), rng=rng)
        a1.append({"file": f, "spk": spk, "ref": librispeech_ref(f),
                   "wav": np.asarray(wav, np.float64), "prot": prot, "arch": arch})
    print(f"A1: {len(a1)} 句", flush=True)

    sl = SensitiveLists.build()
    a2 = pick_a2_files(sl)
    a2_enc = []
    for f, wav, ctc, hits in a2:
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
        spans = ann.merge_spans(margin_frames=8)
        keys = [gen_key() for _ in spans]
        pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
        enc = encrypt_spans(pcm, spans, keys)
        words = {h.word.upper().strip(".,!?;:") for h in hits}
        a2_enc.append({"file": f, "enc": enc, "spans": spans, "keys": keys,
                       "words": words})
    print(f"A2: {len(a2_enc)} 句 / {sum(len(x['words']) for x in a2_enc)} 敏感词",
          flush=True)

    asv = load_asv(device="cuda:0")
    model = load_asr(device="cuda:0", overrides="test_beam_size: 4")
    cache = json.load(open(HYP_CACHE, encoding="utf-8")) if os.path.exists(HYP_CACHE) else {}

    from experiments.eval_asv import embed_wavs

    results = {}
    jobs = []
    for dim, points in SWEEPS.items():
        results[dim] = {"labels": [lab for lab, _ in points],
                        "wer": [], "eer": [], "sim": [], "fnr": []}
        for lab, fn in points:
            key = f"{dim}={lab}"
            # A1：recover(channel(prot))
            recs = [recover(fn(x["prot"]), x["arch"]) for x in a1]
            eer, _ = dataset_eer(asv, [(r, x["spk"]) for r, x in zip(recs, a1)])
            e_org = embed_wavs(asv, [np.asarray(x["wav"], np.float32) for x in a1])
            e_rec = embed_wavs(asv, [np.asarray(r, np.float32) for r in recs])
            sim = float(np.mean(np.sum(e_org * e_rec, axis=1)))
            results[dim]["eer"].append(round(float(eer), 4))
            results[dim]["sim"].append(round(sim, 3))
            for i, r in enumerate(recs):
                jobs.append((f"S9|{key}|recov|{i}", r))
            # A2：decrypt(channel(enc)) 按原帧号
            for i, it in enumerate(a2_enc):
                chan = fn(it["enc"].astype(np.float32) / 32768.0)
                dec = decrypt_spans((np.clip(chan, -1, 1) * 32767).astype(np.int16),
                                    it["spans"], it["keys"])
                jobs.append((f"S9|{key}|dec|{i}", dec.astype(np.float32) / 32768.0))
            print(f"{key}: EER={100*eer:.1f}% sim={sim:.3f}（ASR 待跑）", flush=True)

    print(f"== ASR 共 {len(jobs)} 项 ==", flush=True)
    transcribe_cached(model, jobs, cache)

    for dim, points in SWEEPS.items():
        for lab, _ in points:
            key = f"{dim}={lab}"
            errs = tot = 0
            for i, x in enumerate(a1):
                h = normalize(cache[f"S9|{key}|recov|{i}"])
                r = normalize(x["ref"])
                errs += round(jiwer.wer(r, h) * len(r.split()))
                tot += len(r.split())
            results[dim]["wer"].append(round(100 * errs / tot, 2))
            total = rec = 0
            for i, it in enumerate(a2_enc):
                hyp_tokens = set(normalize(cache[f"S9|{key}|dec|{i}"]).split())
                rec += sum(1 for w in it["words"] if w in hyp_tokens)
                total += len(it["words"])
            results[dim]["fnr"].append(round(1 - rec / total, 4))

    # ── 图：3 面板曲线族 ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150)
    panels = [("bitrate_kbps", "MP3 码率 (kbps)", [32, 64, 128, 192]),
              ("noise_snr_db", "加性噪声 SNR (dB)", [0, 5, 10, 20]),
              ("bandwidth_khz", "重采样带宽 (kHz)", [8.0, 11.025, 22.05])]
    for ax, (dim, xl, xs) in zip(axes, panels):
        r = results[dim]
        ax.plot(xs, r["wer"], "o-", color="#b45536", label="recovered WER %")
        ax.plot(xs, [100 * e for e in r["eer"]], "s--", color="#3a6ea5",
                label="recovered EER %")
        ax2 = ax.twinx()
        ax2.plot(xs, r["sim"], "^-.", color="#4c8c4a", label="sim(orig)")
        ax2.set_ylim(0, 1)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    axes[0].set_ylabel("错误率 %")
    axes[2].set_title("曲线族：信道条件 vs 恢复质量（Conformer/ECAPA）", fontsize=9)
    fig.tight_layout()
    fig_path = os.path.join(RES, "fig_exp9_curves.png")
    fig.savefig(fig_path)
    plt.close(fig)

    json.dump(results, open(os.path.join(RES, "exp9_sweep.json"), "w",
                            encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({d: {k: results[d][k] for k in ("labels", "wer", "fnr")}
                      for d in results}, ensure_ascii=False, indent=1), flush=True)
    print("->", fig_path, flush=True)


if __name__ == "__main__":
    main()
