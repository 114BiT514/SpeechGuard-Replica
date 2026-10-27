"""[B5] VCTK 全集评估：把主表 VCTK 行从 40人×65句 升到论文规模（109人×~75句≈8163）。

论文 Table 4：VCTK = 109 speakers / 8163 utterances / 44.0h。本脚本按"每个可用
说话人取 75 句"逼近该规模（txt 转写缺失的说话人跳过，实际人数在结果中注明）。
协议与 exp3/M3 完全一致：single 模式、dist∈(0.5,0.6)、n∈[1,10]、全对 genuine、
≤1000 随机 impostor、Conformer beam=4、WER 200 句子集。逐说话人流式处理
（44.1k→16k 重采样 + 保护 + 嵌入后即释放），避免 8k×19s 波形驻留内存。

运行（约 1.5-2.5h）：
  E:/Miniconda/envs/speechguard/python.exe experiments/exp7_vctk_full.py
输出：experiments/results/exp3_vctk_full_cache.json（exp3 组装时优先使用）
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

from experiments.eval_asr import corpus_wer, load_asr, normalize  # noqa: E402
from experiments.eval_asv import dataset_eer, embed_wavs, eer_from_scores, load_asv  # noqa: E402
from src.acoustic.warping import protect, recover  # noqa: E402
from experiments.exp3_main_table import _attack_single  # noqa: E402

RES = os.path.join(PROJ, "experiments", "results")
CACHE = os.path.join(RES, "exp3_vctk_full_cache.json")
N_PER_SPK = 75
WER_N = 200


def eer_from_embs(embs, spk, max_imp=1000, seed=0):
    """与 eval_asv.dataset_eer 完全相同的配对协议，作用于预计算嵌入。"""
    rng = np.random.default_rng(seed)
    gen, imp = [], []
    for i in range(len(embs)):
        same = np.where(spk == spk[i])[0]
        same = same[same != i]
        diff = np.where(spk != spk[i])[0]
        if len(same):
            gen.extend(embs[i] @ embs[same].T)
        if len(diff):
            pick = rng.choice(diff, size=min(len(diff), max_imp), replace=False)
            imp.extend(embs[i] @ embs[pick].T)
    return float(eer_from_scores(np.array(gen), np.array(imp)))


def load_speaker_set():
    """返回 [(spk, [flac...])]：txt 与 wav 都有的说话人。"""
    out = []
    for d in sorted(glob.glob(os.path.join(
            PROJ, "data", "vctk", "wav48_silence_trimmed", "p*"))):
        spk = os.path.basename(d)
        if spk in ("s5", "log.txt") or not os.path.isdir(os.path.join(
                PROJ, "data", "vctk", "txt", spk)):
            continue
        fs = sorted(glob.glob(os.path.join(d, "*_mic1.flac")))[:N_PER_SPK]
        if fs:
            out.append((spk, fs))
    return out


def vctk_ref(flac):
    base = os.path.basename(flac).split("_mic")[0]
    tp = os.path.join(PROJ, "data", "vctk", "txt", base.split("_")[0], base + ".txt")
    return normalize(open(tp, encoding="utf-8", errors="replace").readline().strip())


def main():
    import torch
    import torchaudio
    speakers = load_speaker_set()
    total = sum(len(fs) for _, fs in speakers)
    print(f"VCTK 全集：{len(speakers)} 人 / {total} 句（论文 109/8163）", flush=True)
    sel = set(np.random.default_rng(7).choice(total, min(WER_N, total),
                                              replace=False).tolist())
    asv = load_asv("cuda:0")
    asr = load_asr(device="cuda:0", overrides="test_beam_size: 4")

    def load_resample(path):
        w, sr = sf.read(path)
        wt = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
        if sr != 16000:
            wt = torchaudio.functional.resample(wt, sr, 16000)
        return wt.squeeze().numpy().astype(np.float64)

    # 单遍流式：每说话人块 载入+保护一次 → 四态嵌入并行累积
    embs = {s: [] for s in ("ori", "pro", "att", "rec")}
    spk_arr, wsub = [], {s: [] for s in ("ori", "pro", "att", "rec")}
    rsub = []                      # 只收 sel 命中的参考（与 wsub 同遍历序）
    rng_att = np.random.default_rng(2026)
    t0 = time.time()
    gi = 0
    for spk, fs in speakers:
        wavs = [load_resample(f) for f in fs]
        pairs = [protect(w, n=None, dist_range=(0.5, 0.6),
                         rng=np.random.default_rng(3000 + gi + j), mode="single")
                 for j, w in enumerate(wavs)]
        states = {
            "ori": wavs,
            "pro": [p for p, _ in pairs],
            "att": [_attack_single(np.asarray(p, np.float64), rng_att)
                    for p, _ in pairs],
            "rec": [recover(p, a, mode="single") for p, a in pairs],
        }
        for j, f in enumerate(fs):        # WER 子集：选中则存该句四态+参考（对齐）
            if gi + j in sel:
                rsub.append(vctk_ref(f))
                for s, xs in states.items():
                    wsub[s].append(np.asarray(xs[j], np.float32))
        for s, xs in states.items():
            embs[s].append(embed_wavs(asv, [np.asarray(x, np.float64) for x in xs]))
        if gi % (75 * 20) == 0 and gi:
            print(f"  {gi}/{total} ({time.time()-t0:.0f}s)", flush=True)
        spk_arr += [spk] * len(fs)
        gi += len(fs)
        del wavs, pairs, states

    # 缓存 WER 素材（选中 200 句 × 4 态 + 参考），WER 重算不再需要重跑 protect
    def _obj(lst):
        a = np.empty(len(lst), dtype=object)
        for i, x in enumerate(lst):
            a[i] = x
        return a
    np.savez_compressed(os.path.join(RES, "vctk_full_wer_cache.npz"),
                        rsub=_obj(rsub),
                        **{f"w_{s}": _obj(wsub[s]) for s in wsub})

    out_states = {}
    for s in ("ori", "pro", "att", "rec"):
        E = np.concatenate(embs[s])
        out_states[s] = {"eer": round(eer_from_embs(E, np.array(spk_arr)), 4)}
        del E, embs[s]
        torch.cuda.empty_cache()
        print(f"  [{s}] EER={out_states[s]['eer']*100:.2f}% "
              f"({time.time()-t0:.0f}s)", flush=True)

    def _dump():
        m = dict(out_states)
        m["_meta"] = {"n_spk": len(speakers), "n_utts": total,
                      "protocol": "single/dist(0.5,0.6)/n~U[1,10]/ECAPA 1000imp/"
                                  f"Conformer beam4 WER_{WER_N}",
                      "paper_table4": "109 spk / 8163 utts / 44.0h"}
        json.dump(m, open(CACHE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    _dump()                                        # 先落 EER，WER 崩了也不丢

    for s in ("ori", "pro", "att", "rec"):          # 独立一遍 WER（素材已对齐）
        wer, _, _ = corpus_wer(asr, wsub[s], rsub)
        out_states[s]["wer"] = round(float(wer), 4)
        out_states[s]["n_wer"] = len(wsub[s])
        del wsub[s]
        torch.cuda.empty_cache()
        print(f"  [{s}] WER={out_states[s]['wer']*100:.2f}% "
              f"({time.time()-t0:.0f}s)", flush=True)

    out_states["_meta"] = {"n_spk": len(speakers), "n_utts": total,
                           "protocol": "single/dist(0.5,0.6)/n~U[1,10]/ECAPA 1000imp/"
                                       f"Conformer beam4 WER_{WER_N}",
                           "paper_table4": "109 spk / 8163 utts / 44.0h"}
    json.dump(out_states, open(CACHE, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("saved ->", CACHE)


if __name__ == "__main__":
    main()
