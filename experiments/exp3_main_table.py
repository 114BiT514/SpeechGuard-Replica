"""[M5 任务4] 三数据集全系统主表（论文 Table 5/6/7 同款）。

已有数字直接取自 results/*.json（M3/M4 产物）；缺失的 VCTK attacked-WER 与
VCTK recovered EER/WER 在本脚本补跑，协议与 M3 完全一致：
VCTK 40人×65句、single 模式、dist∈(0.5,0.6)、n∈[1,10]、Conformer beam=4。

Volunteers 列：自录集 7 人发音不标准、内容未经人工审听，按委托方要求**不进主表**
（标 hold，论文数字照录供对照）。

运行（GPU 空闲时，约 40-60 分钟）：
  E:/Miniconda/envs/speechguard/python.exe experiments/exp3_main_table.py
输出：experiments/results/main_table.json + main_table.md
"""
from __future__ import annotations

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

RES = os.path.join(PROJ, "experiments", "results")


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


VCTK_CACHE = os.path.join(RES, "exp3_vctk_cache.json")


def build_vctk():
    """{eer:{ori,pro,att,rec}, wer:{...}} —— 供缓存命中与主表组装共用。"""
    c = load("exp3_vctk_cache.json")
    keys = ("ori", "pro", "att", "rec")
    return {"eer": {k: c[k]["eer"] for k in keys},
            "wer": {k: c[k].get("wer") for k in keys}}


def _attack_single(prot_wav, rng):
    from src.acoustic.warping import SegmentParams, _apply_single, gen_params
    a, b, d = gen_params(rng=rng)
    fake = [SegmentParams(0, max(prot_wav.size // 576, 1), a, b, d)]
    out, _ = _apply_single(prot_wav, archive=fake, invert=True)
    return out


def main(vctk_n_utt=65, wer_n=200):
    from experiments.eval_asr import corpus_wer, load_asr, normalize
    from experiments.eval_asv import dataset_eer, load_asv
    from src.acoustic.warping import protect, recover
    import torch
    import torchaudio

    # ── 已有数字 ──
    fte = load("full_testclean_eval.json")
    ls = {
        "eer": {"ori": fte["eer_original"],
                "pro": load("full_testclean_single.json")["eer_protected_single"],
                "att": load("m3_complete.json")["testclean_attacked"]["eer"],
                "rec": load("full_testclean_single_recover.json")["eer_recovered_single"]},
        "wer": {"ori": fte["wer_original"],
                "pro": load("full_testclean_single.json")["wer_protected_single"],
                "att": load("m3_complete.json")["testclean_attacked"]["wer"],
                "rec": load("full_testclean_single_recover.json")["wer_recovered_single"]},
    }
    rtc = load("m3_complete.json")["rtc"]
    fnr = load("fnr_eval.json")

    # ── VCTK 补数：att WER + recover 两态（整段结果缓存，重跑秒出）──
    if os.path.exists(VCTK_CACHE):
        vres = load("exp3_vctk_cache.json")
        print("VCTK 命中缓存，跳过 GPU 评估", flush=True)
    else:
        asv = load_asv("cuda:0")
        asr = load_asr(device="cuda:0", overrides="test_beam_size: 4")
        vdirs = [d for d in sorted(glob.glob(os.path.join(
            PROJ, "data/vctk/wav48_silence_trimmed/p*")))
            if os.path.basename(d) not in ("s5", "log.txt")][:40]
        vfiles, vspk, vrefs, vwav = [], [], [], []
        for d in vdirs:
            for f in sorted(glob.glob(os.path.join(d, "*_mic1.flac")))[:vctk_n_utt]:
                vfiles.append(f)
                vspk.append(os.path.basename(d))
                w, sr = sf.read(f)
                wt = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
                if sr != 16000:
                    wt = torchaudio.functional.resample(wt, sr, 16000)
                vwav.append(wt.squeeze().numpy().astype(np.float64))
                base = os.path.basename(f).split("_mic")[0]
                tp = os.path.join(PROJ, "data/vctk/txt", base.split("_")[0],
                                  base + ".txt")
                vrefs.append(normalize(open(tp, encoding="utf-8", errors="replace")
                                       .readline().strip()))
        print(f"VCTK {len(vwav)} 句加载完成；protect...", flush=True)
        t0 = time.time()
        pairs = [protect(w, n=None, dist_range=(0.5, 0.6),
                         rng=np.random.default_rng(3000 + i), mode="single")
                 for i, w in enumerate(vwav)]
        vprot = [p for p, _ in pairs]
        rng = np.random.default_rng(2026)
        vatt = [_attack_single(p, rng) for p in vprot]
        vrec = [recover(p, a, mode="single") for p, a in pairs]
        print(f"  protect+attack+recover {time.time()-t0:.0f}s", flush=True)

        def eval_c(tag, wavs, do_wer=True):
            eer, _ = dataset_eer(asv, list(zip(wavs, vspk)), max_imp_per_utt=1000)
            rec = {"eer": round(eer, 4)}
            if do_wer:
                idx = np.random.default_rng(7).choice(len(wavs), min(wer_n, len(wavs)),
                                                      replace=False)
                wer, _, _ = corpus_wer(asr, [wavs[i] for i in idx],
                                       [vrefs[i] for i in idx])
                rec["wer"] = round(wer, 4)
            print(f"  [{tag}] EER={eer*100:.2f}%"
                  + (f" WER={rec['wer']*100:.2f}%" if "wer" in rec else ""), flush=True)
            return rec

        vres = {"ori": eval_c("vctk_ori", vwav), "pro": eval_c("vctk_pro", vprot),
                "att": eval_c("vctk_att", vatt), "rec": eval_c("vctk_rec", vrec)}
        json.dump(vres, open(VCTK_CACHE, "w", encoding="utf-8"), indent=2)

    keys = ("ori", "pro", "att", "rec")
    vctk = {"eer": {k: vres[k]["eer"] for k in keys},
            "wer": {k: vres[k]["wer"] for k in keys}}

    out = {
        "table5_style": {
            "LS_test_clean": ls,
            "VCTK": vctk,
            "Volunteers": {"status": "hold：自录集 7 人发音未审听，暂不进主表",
                           "paper": {"eer": [2.21, 30.29, 33.66],
                                     "wer": [6.35, 10.97, 29.99]}},
            "paper_table5": {
                "LS_test_clean": {"eer": [1.47, 34.50, 36.77],
                                  "wer": [2.46, 4.80, 18.39]},
                "VCTK": {"eer": [1.46, 35.66, 37.22], "wer": [6.58, 14.81, 32.69]},
                "recover": {"eer": 7.07, "wer": 9.90}},
        },
        "table6_style_rtc": {
            "ours": rtc,
            "paper": {"SG": [0.32, 0.26], "VoiceMask": [0.29, 0.34],
                      "McAdams": [0.27, 0.21], "Overo": [0.30, 0.31]}},
        "table7_style_fnr": {
            "ours": {"ori_fnr": round(fnr["ori"]["fnr"], 4),
                     "enc_fnr_margin8": round(fnr["enc"]["8"]["fnr"], 4),
                     "dec_fnr": round(fnr["ori"]["fnr"], 4),
                     "n_words": fnr["ori"]["sensitive"]},
            "paper": {"ori": 0.006, "enc": 0.9588, "dec": 0.006}},
    }
    json.dump(out, open(os.path.join(RES, "main_table.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)

    t = out["table5_style"]
    def row(name, d):
        e, w = d["eer"], d["wer"]
        return (f"| {name} | {e['ori']*100:.2f} | {e['pro']*100:.2f} | "
                f"{e['att']*100:.2f} | {w['ori']*100:.2f} | {w['pro']*100:.2f} | "
                f"{w['att']*100:.2f} |")
    md = ["# 主结果表（Table 5 样式）", "",
          "| 数据集 | EER-Ori | EER-Pro | EER-Att | WER-Ori | WER-Pro | WER-Att |",
          "|---|---|---|---|---|---|---|",
          row("LS test-clean（我们）", t["LS_test_clean"]),
          row("VCTK（我们）", t["VCTK"]),
          "| LS test-clean（论文） | 1.47 | 34.50 | 36.77 | 2.46 | 4.80 | 18.39 |",
          "| VCTK（论文） | 1.46 | 35.66 | 37.22 | 6.58 | 14.81 | 32.69 |",
          "",
          f"恢复（EER / WER）：我们 LS {ls['eer']['rec']*100:.2f}% / "
          f"{ls['wer']['rec']*100:.2f}%；我们 VCTK {t['VCTK']['eer']['rec']*100:.2f}% / "
          f"{t['VCTK']['wer']['rec']*100:.2f}%（论文 7.07 / 9.90）",
          f"RTC：protect {rtc['protect']} / recover {rtc['recover']}（论文 0.32/0.26）",
          f"FNR：Enc margin=8 → {fnr['enc']['8']['fnr']*100:.2f}%（论文 95.88%）；"
          f"Dec = Ori（逐字节还原）", ""]
    open(os.path.join(RES, "main_table.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print("saved main_table.json / main_table.md")


if __name__ == "__main__":
    main()
