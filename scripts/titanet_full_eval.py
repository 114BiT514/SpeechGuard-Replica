"""[P1-4] TitaNet 全集复测·第二步：对 titanet_full/ 素材做 TitaNet-L EER（nemo_eval 环境）。

运行：E:/Miniconda/envs/nemo_eval/python.exe scripts/titanet_full_eval.py
输出：experiments/results/titanet_full_titanet.json
结论用途：M3 假说2（"论文高 EER 源于 TitaNet 对扭曲更敏感"）在 24 句集上证伪，
本脚本在 2620 句全集上重验——TitaNet 与 ECAPA 同一素材、同全对配对协议。
"""
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
FULL = os.path.join(PROJ, "data", "probe", "titanet_full")


def embed_all(m, paths, tag):
    embs, t0 = [], time.time()
    for i, p in enumerate(paths):
        out = m.infer_file(p)
        e = out[1] if isinstance(out, tuple) and len(out) > 1 else out
        embs.append(np.asarray(e).ravel())
        if i % 400 == 399:
            print(f"  [{tag}] {i+1}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
    e = np.stack(embs)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def pair_eer(embs, spk, max_imp=1000):
    from experiments.eval_asv import eer_from_scores
    rng = np.random.default_rng(0)
    gen, imp = [], []
    for i in range(len(embs)):
        same = np.where(spk == spk[i])[0]
        same = same[same != i]
        gen.extend(embs[i] @ embs[same].T)
        diff = np.where(spk != spk[i])[0]
        pick = rng.choice(diff, size=min(max_imp, len(diff)), replace=False)
        imp.extend(embs[i] @ embs[pick].T)
    return float(eer_from_scores(np.array(gen), np.array(imp)))


def main():
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel as M
    m = M.from_pretrained("nvidia/speakerverification_en_titanet_large")
    m.eval()
    with open(os.path.join(FULL, "spk.txt")) as fh:
        spk = np.array(fh.read().split())
    o = sorted(os.path.join(FULL, "original", x)
               for x in os.listdir(os.path.join(FULL, "original")))
    p = sorted(os.path.join(FULL, "protected", x)
               for x in os.listdir(os.path.join(FULL, "protected")))
    assert len(o) == len(spk) == len(p), (len(o), len(spk), len(p))
    eo = pair_eer(embed_all(m, o, "orig"), spk)
    ep = pair_eer(embed_all(m, p, "prot"), spk)
    ecapa = json.load(open(os.path.join(PROJ, "experiments", "results",
                                         "titanet_full_ecapa.json")))
    out = {"titanet_eer_original": round(eo, 4),
           "titanet_eer_protected": round(ep, 4),
           "ecapa_reference": ecapa,
           "hypothesis2_verdict": (
               "TitaNet 保护后 EER 仍显著低于 ECAPA → 假说2（TitaNet 更敏感）在全集上"
               "同样证伪" if ep < ecapa["ecapa_eer_protected"] else
               "TitaNet 全集保护后 EER ≥ ECAPA → 假说2 部分成立，需修订 M3 结论"),
           "n_utts": len(o)}
    json.dump(out, open(os.path.join(PROJ, "experiments", "results",
                                     "titanet_full_titanet.json"), "w"), indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
