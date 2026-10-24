"""用 NVIDIA TitaNet-Large（论文 Fig.7 的 ASV 模型）复测保护音频的 EER。

目的：检验假说 2——论文的高 EER 是否来自 TitaNet 对频域扭曲更敏感，
而非 ECAPA（我们主评估用的模型）能感知到的匿名化强度。

在独立环境 nemo_eval 中运行（不污染主环境）：
    E:/Miniconda/envs/nemo_eval/python.exe scripts/titanet_eval.py
需要网络（HF 下载 nvidia/titanet-l，可设 HF_ENDPOINT/代理）。
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from experiments.eval_asv import eer_from_scores  # noqa: E402

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"


def load_model():
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel as EncDecSpeakerModel

    try:
        m = EncDecSpeakerModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large")
    except Exception as e:  # noqa: BLE001
        print("from_pretrained titanet_large failed:", e, flush=True)
        m = EncDecSpeakerModel.from_pretrained("nvidia/speakerverification_en_titanet")
    m.eval()
    return m, torch


def embed_batch(m, torch, files):
    """用 NeMo infer_file 逐文件提取嵌入（内部完成 mel 预处理）。"""
    embs = []
    for p in files:
        out = m.infer_file(p)
        e = out[1] if isinstance(out, tuple) and len(out) > 1 else out
        e = np.asarray(e).ravel()
        embs.append(e)
    e = np.stack(embs)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def pair_eer(m, torch, files):
    embs = embed_batch(m, torch, files)
    with open(os.path.join(PROJ, "data", "probe", "spk.txt")) as f:
        spk = np.array(f.read().split())
    gen, imp = [], []
    rng = np.random.default_rng(0)
    for i in range(len(files)):
        same = np.where(spk == spk[i])[0]
        same = same[same != i]
        diff = np.where(spk != spk[i])[0]
        gen.extend(embs[i] @ embs[same].T)
        pick = rng.choice(diff, size=min(len(diff), 15), replace=False)
        imp.extend(embs[i] @ embs[pick].T)
    return eer_from_scores(gen, imp)


def main():
    m, torch = load_model()
    orig = sorted(os.listdir(os.path.join(PROJ, "data", "probe", "original")))
    prot = sorted(os.listdir(os.path.join(PROJ, "data", "probe", "protected_block")))
    o = [os.path.join(PROJ, "data", "probe", "original", x) for x in orig]
    p = [os.path.join(PROJ, "data", "probe", "protected_block", x) for x in prot]
    eer_o = pair_eer(m, torch, o)
    eer_p = pair_eer(m, torch, p)
    print(f"TitaNet-Large: 原始 EER={eer_o*100:.2f}% | 块级保护后 EER={eer_p*100:.2f}%"
          f"（ECAPA 同音频 37.5%，论文 40.09%）", flush=True)


if __name__ == "__main__":
    main()
