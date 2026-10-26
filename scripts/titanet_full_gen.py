"""[P1-4] TitaNet 全集复测·第一步：生成 test-clean 全集 single 模式保护音频 + ECAPA EER。

M3 findings §4 自留 TODO：假说2 的 TitaNet 检验当时用 24 句子集（小子集失真，
M3 教训 §2），须全集重做。本脚本产出两份环境共享的素材：
  data/probe/titanet_full/original/*.wav
  data/probe/titanet_full/protected/*.wav   （single 模式，dist∈(0.5,0.6)，n∈[1,10]）
  data/probe/titanet_full/spk.txt           （说话人标签，按文件名序）
  experiments/results/titanet_full_ecapa.json（ECAPA 对照，本次现算）
第二步（TitaNet 推理）：E:/Miniconda/envs/nemo_eval/python.exe scripts/titanet_full_eval.py

运行：E:/Miniconda/envs/speechguard/python.exe scripts/titanet_full_gen.py
"""
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from src.acoustic.warping import protect  # noqa: E402

FULL = os.path.join(PROJ, "data", "probe", "titanet_full")


def main():
    os.makedirs(os.path.join(FULL, "original"), exist_ok=True)
    os.makedirs(os.path.join(FULL, "protected"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(
        PROJ, "data", "librispeech", "LibriSpeech", "test-clean", "*", "*", "*.flac")))
    rng = np.random.default_rng(2026)
    spk = []
    t0 = time.time()
    for i, f in enumerate(files):
        w, sr = sf.read(f)
        w = np.asarray(w, np.float64)
        base = os.path.splitext(os.path.basename(f))[0]
        uid = base.split("-")[0]
        p, _arch = protect(w, n=None, dist_range=(0.5, 0.6), rng=rng, mode="single")
        sf.write(os.path.join(FULL, "original", f"{i:05d}_{base}.wav"), w, sr)
        sf.write(os.path.join(FULL, "protected", f"{i:05d}_{base}.wav"),
                 np.asarray(p, np.float64), sr)
        spk.append(uid)
        if i % 400 == 399:
            print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(FULL, "spk.txt"), "w") as fh:
        fh.write("\n".join(spk))
    print(f"生成完成 {len(files)} 对 / {time.time()-t0:.0f}s", flush=True)

    # ECAPA 对照（本次现算，与 TitaNet 完全同一素材）
    from experiments.eval_asv import dataset_eer, load_asv
    asv = load_asv("cuda:0")
    eo, _ = dataset_eer(asv, [(w, s) for w, s in
                              zip([sf.read(p)[0] for p in sorted(
                                  glob.glob(os.path.join(FULL, "original", "*.wav")))], spk)],
                        max_imp_per_utt=1000)
    ep, _ = dataset_eer(asv, [(w, s) for w, s in
                              zip([sf.read(p)[0] for p in sorted(
                                  glob.glob(os.path.join(FULL, "protected", "*.wav")))], spk)],
                        max_imp_per_utt=1000)
    out = {"n_utts": len(files), "n_spk": len(set(spk)),
           "ecapa_eer_original": round(float(eo), 4),
           "ecapa_eer_protected": round(float(ep), 4)}
    json.dump(out, open(os.path.join(PROJ, "experiments", "results",
                                     "titanet_full_ecapa.json"), "w"), indent=2)
    print("ECAPA 全集：", out)


if __name__ == "__main__":
    main()
