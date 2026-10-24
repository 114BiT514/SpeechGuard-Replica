"""[M3 任务2] ASV（ECAPA-TDNN）评估工具：嵌入、余弦分数、EER。

论文 §5.1 协议：ECAPA-TDNN（VoxCeleb 训练）提说话人嵌入，余弦相似度打分，
EER 衡量匿名性（越高越匿名，0.5 为最优）。论文选参实验用的 ASV 是
TitaNet-Large、评估用 ECAPA-TDNN；本文统一用 ECAPA-TDNN，模型差异在报告注明。
"""
from __future__ import annotations

import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)


def load_asv(device: str = "cuda:0", savedir: str | None = None):
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    savedir = savedir or os.path.join(PROJ, "data", "pretrained", "ecapa-voxceleb")
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        run_opts={"device": device},
        local_strategy=(
            LocalStrategy.COPY_SKIP_CACHE if sys.platform == "win32" else LocalStrategy.SYMLINK
        ),
    )


def embed_wavs(model, wavs: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
    """16kHz mono 波形列表 → (N, 192) L2 归一化嵌入。"""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(wavs), batch_size):
            batch = wavs[i : i + batch_size]
            lens = torch.tensor([len(w) for w in batch], dtype=torch.float32)
            max_len = int(lens.max())
            mat = torch.zeros(len(batch), max_len)
            for j, w in enumerate(batch):
                mat[j, : len(w)] = torch.as_tensor(np.asarray(w, np.float32))
            rel = lens / max_len
            embs = model.encode_batch(mat.to(model.device), rel.to(model.device))
            out.append(embs.squeeze(1).cpu().numpy())
    embs = np.concatenate(out, axis=0)
    return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)


def eer_from_scores(gen_scores, imp_scores) -> float:
    """余弦分数 → EER（降序扫描 + 交叉点线性插值）。完美可分→0，同分布→0.5。"""
    gen_scores = np.asarray(gen_scores, dtype=np.float64)
    imp_scores = np.asarray(imp_scores, dtype=np.float64)
    n_gen, n_imp = len(gen_scores), len(imp_scores)
    if n_gen == 0 or n_imp == 0:
        return float("nan")
    scores = np.concatenate([gen_scores, imp_scores])
    labels = np.concatenate([np.ones(n_gen), np.zeros(n_imp)])
    order = np.argsort(-scores)
    lab = labels[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(1.0 - lab)
    far = fp / n_imp                    # 阈值取第 k 名分数时 impostor 被接受率
    frr = (n_gen - tp) / n_gen          # genuine 被拒绝率
    # 降序扫描：k=1（只收最高分）时 FAR≈0/FRR≈1（d<0），k=M（全收）时 FAR=1/FRR=0（d>0），
    # EER 在 d 由负转正的穿越点
    d = far - frr
    neg = np.argwhere(d < 0)
    if len(neg) == 0:
        return float(far[0])
    j = int(neg[-1])                    # 最后一个 d<0 的位置，与 d[j+1] 之间穿越 0
    if j + 1 >= len(d):
        return float((far[j] + frr[j]) / 2)
    denom = d[j + 1] - d[j]
    t = (0.0 - d[j]) / denom if denom != 0 else 0.0
    return float(far[j] + t * (far[j + 1] - far[j]))


def dataset_eer(model, items, max_imp_per_utt: int = 15, seed: int = 0) -> tuple[float, np.ndarray]:
    """items: [(wav_path | np.ndarray, speaker_id)] → (EER, embeddings)。

    genuine：同说话人不同语音的全部配对；impostor：跨说话人随机配对
    （每条语音最多 max_imp_per_utt 条，控制规模）。
    """
    import soundfile as sf

    files = [it[0] for it in items]
    spk = np.array([it[1] for it in items])
    wavs = [it[0] if isinstance(it[0], np.ndarray) else sf.read(it[0])[0] for it in items]
    wavs = [np.asarray(w, np.float64).ravel() for w in wavs]
    embs = embed_wavs(model, wavs)
    rng = np.random.default_rng(seed)
    n = len(items)
    gen_scores, imp_scores = [], []
    for i in range(n):
        same = np.where(spk == spk[i])[0]
        same = same[same != i]
        diff = np.where(spk != spk[i])[0]
        if len(same):
            gen_scores.extend(embs[i] @ embs[same].T)
        if len(diff):
            pick = rng.choice(diff, size=min(len(diff), max_imp_per_utt), replace=False)
            imp_scores.extend(embs[i] @ embs[pick].T)
    return eer_from_scores(gen_scores, imp_scores), embs
