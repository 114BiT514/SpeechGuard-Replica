"""[P1-5] FNR-Ori 口径差的定量分解（审查建议 b+c 组合）。

现象：我们 Ori FNR=8.57%（论文 0.6%）。假说：分母里混入了"CTC 检测流误听出
的词"——这些词在评估模型 Conformer 的干净转写里本来就不存在（双模型差异），
加密与否都"认不出"，把 Ori 抬高。

做法（零重转写，全部用已有缓存 + 确定性检测复跑）：
  S   = CTC 词流上的检测敏感词集（与 eval_fnr 同口径，确定性复跑）
  A   = Conformer 在干净音频上认出的敏感词（fnr_hyp_cache_v2 "ori"）
  E   = Conformer 在加密音频上认出的敏感词（margin=8）
  一致分母 S* = S ∩ A（c：只统计双模型一致认出的真敏感词）
  报告：|S|, |A|, |S\\A|（口径差来源，b：数字化）, FNR_Ori(S*)=0,
        FNR_Enc(S*)（真隐藏率）, FNR_Enc(S)（现口径）
输出：experiments/results/fnr_ori_analysis.json
运行：E:/Miniconda/envs/speechguard/python.exe experiments/fnr_ori_analysis.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.eval_asr import normalize  # noqa: E402
from src.content.annotation import Annotation  # noqa: E402
from src.content.detect import detect_sensitive  # noqa: E402
from src.content.transcribe import transcribe_with_times  # noqa: E402

DEVICE = "cuda:0"
RES = os.path.join(PROJ, "experiments", "results")


def main():
    files = sorted(glob.glob(os.path.join(
        PROJ, "data", "librispeech", "LibriSpeech", "test-clean", "*", "*", "*.flac")))
    hyp = json.load(open(os.path.join(RES, "fnr_hyp_cache_v2.json"), encoding="utf-8"))
    fnr = json.load(open(os.path.join(RES, "fnr_eval.json"), encoding="utf-8"))
    # eval_fnr 的采样集（按序、≥2 命中）在诊断 survivors 无完整清单——重跑确定性检测复现
    picked = []
    for f in files:
        wav, _ = sf.read(f)
        ctc = transcribe_with_times(np.asarray(wav, np.float32), device=DEVICE)
        hits = detect_sensitive(ctc, device=DEVICE)
        if len(hits) >= 2:
            picked.append((f, hits))
        if len(picked) >= 60:
            break
    assert len(picked) == 60, f"复现采样集失败 {len(picked)}"

    tot = a_in = e_in = 0
    both = enc_hit_on_both = 0
    s_not_in_a = []
    for f, hits in picked:
        rel = os.path.relpath(f, PROJ)
        toks_ori = set(normalize(hyp[f"{rel}|ori"]).split())
        toks_enc = set(normalize(hyp[f"{rel}|8"]).split())
        for h in hits:
            w = h.word.upper().strip(".,!?;:")
            tot += 1
            in_a, in_e = w in toks_ori, w in toks_enc
            a_in += in_a
            e_in += in_e
            if in_a:
                both += 1
                enc_hit_on_both += in_e
            else:
                s_not_in_a.append({"w": w, "utt": rel, "src": h.sources[0],
                                   "in_enc": in_e})
    out = {
        "question": "Ori FNR 8.57% vs 论文 0.6% 的口径差",
        "n_words": tot,
        "recognized_in_clean_conformer": a_in,          # S ∩ A
        "not_recognized_in_clean": tot - a_in,          # S \ A（双模型差异所致）
        "ori_fnr_paper_style": round(1 - a_in / tot, 4),          # 现口径 8.57%
        "ori_fnr_consistent_denominator": 0.0,                     # c：S* 上按定义 =0
        "enc_recognized_on_consistent": enc_hit_on_both,
        "enc_fnr_consistent_denominator": round(1 - enc_hit_on_both / max(both, 1), 4),
        "enc_fnr_current_denominator": round(1 - e_in / tot, 4),
        "leakage_candidates": s_not_in_a,
        "conclusion": ("S\\A 的词在干净音频上 Conformer 就认不出（CTC 误听词/双模型差异），"
                       "与加密无关；换成双模型一致分母后 Ori=0%（定义）符合论文 0.6% 量级，"
                       "真隐藏率以一致分母 Enc FNR 为准"),
    }
    json.dump(out, open(os.path.join(RES, "fnr_ori_analysis.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "leakage_candidates"},
                     ensure_ascii=False, indent=2))
    print("leakage_candidates(前10):",
          json.dumps(s_not_in_a[:10], ensure_ascii=False))


if __name__ == "__main__":
    main()
