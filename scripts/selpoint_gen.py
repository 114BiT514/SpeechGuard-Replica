"""[B6] 论文选参工作点复现·第 1 步：dev-clean n=1、dist∈(0.5,0.6) 保护素材 + ECAPA 对照。

论文 §4.2/Table 2 语境：选参实验用 ASR=SpeechBrain Transformer、ASV=NVIDIA
TitaNet-Large，工作点报 "ASV 错误率 80.18%、ASR 错误率 3.58%"。
本步关键推断（可由论文自身数字验证）：80.18 = 2×40.09 —— 论文 Table 2/Fig.7
在 (0.5,0.6) 档的 EER 恰为 40.09%，而 EER 交叉点处 FAR=FRR=EER，
故 "ASV 错误率" = FAR+FRR = 2×EER。此推断写入结果 json，第 2 步用 TitaNet 实测。

运行：E:/Miniconda/envs/speechguard/python.exe scripts/selpoint_gen.py
输出：data/probe/selpoint/{original,protected}/ + spk.txt
     experiments/results/selpoint_ecapa.json
"""
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from experiments.eval_asv import dataset_eer, load_asv  # noqa: E402
from src.acoustic.warping import protect  # noqa: E402

OUT = os.path.join(PROJ, "data", "probe", "selpoint")


def main():
    os.makedirs(os.path.join(OUT, "original"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "protected"), exist_ok=True)
    wavs, spk = [], []
    for d in sorted(glob.glob(os.path.join(
            PROJ, "data", "librispeech", "LibriSpeech", "dev-clean", "*")))[:25]:
        for f in sorted(glob.glob(os.path.join(d, "*", "*.flac")))[:40]:
            wavs.append(np.asarray(sf.read(f)[0], np.float64))
            spk.append(os.path.basename(os.path.dirname(os.path.dirname(f))))
    print(f"dev-clean 工作点集：{len(wavs)} 句 / {len(set(spk))} 人（n=1, dist(0.5,0.6)）",
          flush=True)
    rng = np.random.default_rng(2026)
    prot = [protect(w, n=1, dist_range=(0.5, 0.6), rng=rng, mode="single")[0]
            for w in wavs]
    with open(os.path.join(OUT, "spk.txt"), "w") as fh:
        fh.write("\n".join(spk))
    for i, (a, b) in enumerate(zip(wavs, prot)):
        sf.write(os.path.join(OUT, "original", f"{i:05d}.wav"), a, 16000)
        sf.write(os.path.join(OUT, "protected", f"{i:05d}.wav"), b, 16000)

    asv = load_asv("cuda:0")
    eo, _ = dataset_eer(asv, list(zip(wavs, spk)), max_imp_per_utt=1000)
    ep, _ = dataset_eer(asv, list(zip(prot, spk)), max_imp_per_utt=1000)
    out = {
        "protocol": "dev-clean 25×40, single 模式, n=1, dist∈(0.5,0.6), ECAPA 全集协议",
        "ecapa_eer_original": round(float(eo), 4),
        "ecapa_eer_protected": round(float(ep), 4),
        "paper": {"eer_0_6": 0.4009, "asv_err_rate": 0.8018,
                  "identity_check_2xEER": abs(2 * 0.4009 - 0.8018) < 1e-9,
                  "asr_err_rate": 0.0358,
                  "note": "论文 ASV'错误率'=FAR+FRR@EER 交叉点=2×EER（40.09×2=80.18，"
                          "与论文自身 Table 2 数字精确自洽）；ASR 3.58% 用 SpeechBrain "
                          "Transformer，我们 fig7_single 同档 Conformer WER=4.88%（模型口径差）"},
        "ours_far_frr_ecapa": round(2 * float(ep), 4)}
    json.dump(out, open(os.path.join(PROJ, "experiments", "results",
                                     "selpoint_ecapa.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
