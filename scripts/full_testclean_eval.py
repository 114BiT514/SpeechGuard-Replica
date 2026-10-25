"""[M3 关键实验] 完整 test-clean（40人×2620条）上的 EER/WER——检验"评估集规模"假说。

动机：此前所有 EER 测量在 8×3=24 句小子集上，genuine 对仅 24 个（EER 分辨率~4%）。
论文 Table 5 在完整 test-clean 上报告：原始 EER 1.47% / 保护后 34.50%；
WER 原始 2.46% / 保护后 4.80%（n∈[1,10] 随机，dist(0.5,0.6)）。
小子集可能系统性低估保护后 EER（每句随机扭曲参数 → 说话人内变异需大样本才显现）。

协议：全部 utterance 两两配对（genuine=同说话人，impostor=异说话人），
ECAPA-TDNN 嵌入余弦相似度 → EER。WER 用 300 句随机子集（beam=4，全集太慢）。

产出：experiments/results/full_testclean_eval.json
运行：E:/Miniconda/envs/speechguard/python.exe scripts/full_testclean_eval.py
"""
import glob
import json
import os
import sys
import time

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import soundfile as sf  # noqa: E402

from experiments.eval_asr import corpus_wer, librispeech_ref, load_asr  # noqa: E402
from experiments.eval_asv import dataset_eer, load_asv  # noqa: E402
from src.acoustic.warping import protect  # noqa: E402

SR = 16000
WER_SUBSET = 300


def main():
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(
        PROJ, "data/librispeech/LibriSpeech/test-clean/*/*/*.flac")))
    spk = [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in files]
    print(f"test-clean: {len(files)} utts, {len(set(spk))} speakers", flush=True)

    wavs = [sf.read(f)[0].astype(np.float64) for f in files]

    print("加载 ECAPA...", flush=True)
    asv = load_asv("cuda:0")
    print("加载 Conformer...", flush=True)
    asr = load_asr("cuda:0", overrides="test_beam_size: 4")

    out = {"n_utts": len(files), "n_spk": len(set(spk))}

    # ---- 原始 EER（全对协议）----
    eer_o, _ = dataset_eer(asv, list(zip(wavs, spk)), max_imp_per_utt=1000)
    print(f"[原始] EER={eer_o*100:.2f}% (论文 1.47%) | {time.time()-t0:.0f}s", flush=True)
    out["eer_original"] = round(eer_o, 4)

    # ---- 保护（n∈[1,10] 随机 per segment，论文原型 §5.1 设置）----
    rng = np.random.default_rng(2026)
    prot = []
    for i, w in enumerate(wavs):
        p, _ = protect(w, n=None, dist_range=(0.5, 0.6), rng=rng)
        prot.append(p)
        if (i + 1) % 500 == 0:
            print(f"  protect {i+1}/{len(wavs)} ({time.time()-t0:.0f}s)", flush=True)

    eer_p, _ = dataset_eer(asv, list(zip(prot, spk)), max_imp_per_utt=1000)
    print(f"[保护] EER={eer_p*100:.2f}% (论文 34.50%) | {time.time()-t0:.0f}s", flush=True)
    out["eer_protected"] = round(eer_p, 4)

    # ---- WER 子集 ----
    sel = np.random.default_rng(7).choice(len(files), WER_SUBSET, replace=False)
    refs = [librispeech_ref(files[i]) for i in sel]
    wer_o, _, _ = corpus_wer(asr, [wavs[i] for i in sel], refs)
    print(f"[原始] WER={wer_o*100:.2f}% on {WER_SUBSET} utts (论文全集 2.46%) | {time.time()-t0:.0f}s", flush=True)
    wer_p, _, _ = corpus_wer(asr, [prot[i] for i in sel], refs)
    print(f"[保护] WER={wer_p*100:.2f}% on {WER_SUBSET} utts (论文 4.80%) | {time.time()-t0:.0f}s", flush=True)
    out["wer_original"] = round(wer_o, 4)
    out["wer_protected"] = round(wer_p, 4)
    out["wer_subset"] = int(WER_SUBSET)
    out["elapsed_s"] = round(time.time() - t0, 1)

    dst = os.path.join(PROJ, "experiments", "results", "full_testclean_eval.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved:", dst, flush=True)
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
