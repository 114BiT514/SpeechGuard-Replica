"""[B6] 论文选参工作点复现·第 2 步：TitaNet-Large 在 dev-clean n=1 (0.5,0.6) 素材上的 EER
（论文选参 ASV 模型本体），验证 "ASV 错误率 80.18%"。

素材由 scripts/selpoint_gen.py 生成。运行（nemo_eval 环境，代理在线）：
  HTTPS_PROXY=http://127.0.0.1:7890 E:/Miniconda/envs/nemo_eval/python.exe scripts/titanet_selpoint.py
输出：experiments/results/selpoint_titanet.json
"""
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
FULL = os.path.join(PROJ, "data", "probe", "selpoint")


def embed_all(m, paths, tag):
    embs = []
    for i, p in enumerate(paths):
        out = m.infer_file(p)
        e = out[1] if isinstance(out, tuple) and len(out) > 1 else out
        embs.append(np.asarray(e).ravel())
        if i % 200 == 199:
            print(f"  [{tag}] {i+1}/{len(paths)}", flush=True)
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
    o = sorted(glob.glob(os.path.join(FULL, "original", "*.wav")))
    p = sorted(glob.glob(os.path.join(FULL, "protected", "*.wav")))
    eo = pair_eer(embed_all(m, o, "orig"), spk)
    ep = pair_eer(embed_all(m, p, "prot"), spk)
    ecapa = json.load(open(os.path.join(PROJ, "experiments", "results",
                                        "selpoint_ecapa.json")))
    out = {
        "protocol": "同 selpoint_ecapa（dev-clean 25×40, n=1, dist(0.5,0.6), single）",
        "titanet_eer_original": round(eo, 4),
        "titanet_eer_protected": round(ep, 4),
        "titanet_far_frr_at_eer": round(2 * ep, 4),
        "paper_asv_err_rate": 0.8018,
        "verdict": (f"TitaNet FAR+FRR={2*ep:.2%} vs 论文 80.18% → "
                    + ("一致（±3pp 内）" if abs(2 * ep - 0.8018) <= 0.03 else
                       f"偏差 {abs(2*ep-0.8018):.1%}（single 模式与论文逐段模式差异，"
                       "结合 2×EER 口径推断见 selpoint_ecapa.json）")),
        "ecapa_reference": ecapa}
    json.dump(out, open(os.path.join(PROJ, "experiments", "results",
                                     "selpoint_titanet.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "ecapa_reference"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
