"""[M4 任务8] 敏感词加密 FNR 完整评估（论文 §5.1 协议 + Table 7 口径）。

流程与论文一致：
  转写+词时间戳（wav2vec2 CTC，代替 WavLM+MFA，见 README 模型替换说明）
  → 自动检测（词表+模式+NER，src/content/detect.py）→ 默认标注
  → 帧级 AES-CTR 加密 → **Conformer ASR**（论文评估协议指定模型）重新转写
  → FNR = 加密后 ASR 认不出的敏感词比例（论文口径：Enc≈95.9%、Ori≈0.6%、Dec≈0.6%）。

与 scripts/m4_e2e_test.py（6 句冒烟、参考文本对齐、FNR 75%）的区别：
  1) 检测直接作用于 CTC 词流（时间戳与词同源，消除 difflib 插值误差——冒烟版
     的失配主要来自插值时间戳）；
  2) 评估 ASR 用 Conformer（论文协议），wav2vec2 只负责词时间戳；
  3) 余量扫描 {4,8,12,16} 帧，定位 FNR 损失来源；
  4) 样本量 60 句（含 ≥2 个敏感词的 test-clean 语音），逐字节解密还原校验。

运行（约 10-20 分钟 GPU）：
  E:/Miniconda/envs/speechguard/python.exe experiments/eval_fnr.py
输出：experiments/results/fnr_eval.json + 控制台表格
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

from experiments.eval_asr import (  # noqa: E402
    librispeech_ref, load_asr, normalize, transcribe)
from src.content.annotation import Annotation  # noqa: E402
from src.content.detect import detect_sensitive  # noqa: E402
from src.content.encrypt import decrypt_spans, encrypt_spans, gen_key  # noqa: E402
from src.content.sensitive_lists import SensitiveLists  # noqa: E402
from src.content.transcribe import transcribe_with_times  # noqa: E402

SR = 16000
N_UTTS = 60
MARGINS = [4, 8, 12, 16]
DEVICE = "cuda:0"


def group_consecutive(idxs: list[int]) -> list[list[int]]:
    groups, cur = [], [idxs[0]]
    for i in idxs[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups


def main():
    files = sorted(glob.glob(os.path.join(
        PROJ, "data", "librispeech", "LibriSpeech", "test-clean", "*", "*", "*.flac")))
    sl = SensitiveLists.build()

    # ── 采样：找含 ≥2 个敏感词的语音 ──
    print("扫描 test-clean 采样（CTC 转写 + 自动检测）...", flush=True)
    picked, t0 = [], time.time()
    scanned = 0
    for f in files:
        wav, _ = sf.read(f)
        wav = wav.astype(np.float32)
        ctc = transcribe_with_times(wav, device=DEVICE)
        hits = detect_sensitive(ctc, sl, device=DEVICE)
        scanned += 1
        if len(hits) >= 2:
            picked.append((f, wav, ctc, hits))
        if len(picked) >= N_UTTS:
            break
    print(f"采样 {len(picked)} 句（扫描 {scanned} 句）/ 用时 {time.time() - t0:.0f}s "
          f"，敏感词共 {sum(len(h) for *_, h in picked)} 个", flush=True)

    # ── 构造标注 + 各余量密文 ──
    jobs = []                      # (rel, margin|"ori", wav_float)
    metas = []
    for f, wav, ctc, hits in picked:
        ref = librispeech_ref(f)
        rel = os.path.relpath(f, PROJ)
        ann = Annotation.from_ctc(f, ctc)
        # 相邻命中词合并为一个加密对象（论文：每个加密对象独立密钥 w_j）
        for g in group_consecutive(sorted(h.idx for h in hits)):
            by_idx = {h.idx: h for h in hits}
            ann.mark_words(g, by_idx[g[0]].sources[0])

        pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
        sensitive = {}
        for h in hits:
            sensitive[h.word.upper().strip(".,!?;:")] = {
                "w": h.word, "t0": round(h.t0, 3), "t1": round(h.t1, 3),
                "dur": round(h.t1 - h.t0, 3), "src": h.sources[0]}
        meta = {"file": os.path.relpath(f, PROJ), "ref": ref,
                "words": list(sensitive.values())}
        for m in MARGINS:
            spans = ann.merge_spans(margin_frames=m)
            keys = [gen_key() for _ in spans]
            enc = encrypt_spans(pcm, spans, keys)
            assert np.array_equal(decrypt_spans(enc, spans, keys), pcm), "解密还原失败"
            jobs.append((rel, m, enc.astype(np.float32) / 32768.0))
        jobs.append((rel, "ori", wav))
        meta["cov_frames"] = {str(m): sum(fe - fs + 1
                                         for fs, fe in ann.merge_spans(margin_frames=m))
                              for m in MARGINS}
        meta["total_frames"] = int(np.ceil(len(pcm) / 576))
        metas.append(meta)

    # ── Conformer 批量转写（论文评估协议 ASR），带磁盘缓存（GPU 结果可复用）──
    # 缓存版本：v2 = 检测收紧后（虚词过滤 + date 正则收窄），与 v1 密文不可混用
    cache_path = os.path.join(PROJ, "experiments", "results", "fnr_hyp_cache_v2.json")
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
    todo = [(r, m, w) for r, m, w in jobs if f"{r}|{m}" not in cache]
    print(f"缓存命中 {len(jobs) - len(todo)}/{len(jobs)}", flush=True)
    if todo:
        print("加载 Conformer（test_beam_size=4，与 M3 WER 评估同配置）...", flush=True)
        model = load_asr(device=DEVICE, overrides="test_beam_size: 4")
        print(f"转写 {len(todo)} 条音频 ...", flush=True)
        t0 = time.time()
        hyps = []
        CH = 30
        for s in range(0, len(todo), CH):
            hyps += transcribe(model, [w for _, _, w in todo[s:s + CH]], batch_size=2)
            el = time.time() - t0
            print(f"  {len(hyps)}/{len(todo)}（{el:.0f}s，"
                  f"{el / len(hyps) * (len(todo) - len(hyps)):.0f}s 后完成）", flush=True)
        for (r, m, _), hyp in zip(todo, hyps):
            cache[f"{r}|{m}"] = hyp
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)

    by_utt: dict[str, dict] = {}
    for r, m, _ in jobs:
        by_utt.setdefault(r, {})[m] = cache[f"{r}|{m}"]

    def score(key):
        """返回 (敏感词总数, 仍被识别数, 幸存词列表)。"""
        total = rec = 0
        survivors = []
        for meta in metas:
            hyp_tokens = set(normalize(by_utt[meta["file"]][key]).split())
            for w in meta["words"]:
                total += 1
                if w["w"].upper().strip(".,!?;:") in hyp_tokens:
                    rec += 1
                    survivors.append({**w, "utt": meta["file"]})
        return total, rec, survivors

    results = {"protocol": {
        "dataset": "LibriSpeech test-clean", "n_utts": len(metas),
        "asr_eval": "speechbrain/asr-conformer-transformerlm-librispeech",
        "timestamps": "wav2vec2-large-960h-lv60-self CTC",
        "margins_frames": MARGINS,
        "fnr_def": "加密后 ASR 未识别出的敏感词比例（论文 Table 7 口径，越高越私密）"},
        "ori": {}, "enc": {}, "diagnostics": {}}

    st, rec, _ = score("ori")
    results["ori"] = {"sensitive": st, "recognized": rec, "fnr": 1 - rec / st}
    print(f"Ori: 敏感词 {st}，ASR 识别 {rec}，FNR={1 - rec / st:.2%}（论文 ≈0.6%）")

    import jiwer
    for m in MARGINS:
        st, rec, surv = score(m)
        wers = [jiwer.wer(normalize(meta["ref"]), normalize(by_utt[meta["file"]][m]))
                for meta in metas]
        cov = np.mean([meta["cov_frames"][str(m)] / meta["total_frames"] for meta in metas])
        results["enc"][str(m)] = {
            "sensitive": st, "recognized": rec, "fnr": 1 - rec / st,
            "wer_overall": float(np.mean(wers)), "frame_coverage": float(cov)}
        results["diagnostics"][str(m)] = surv
        print(f"Enc margin={m:2d}帧: FNR={1 - rec / st:.2%}（论文 Enc≈95.88%）"
              f" 仍识别 {rec}/{st}  整句WER={np.mean(wers):.2%}"
              f"  加密帧占比={cov:.1%}", flush=True)

    out = os.path.join(PROJ, "experiments", "results", "fnr_eval.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print("saved ->", out)

    surv = results["diagnostics"][str(MARGINS[-1])]
    if surv:
        from collections import Counter
        print(f"\n[诊断] margin={MARGINS[-1]} 仍被识别的词（共 {len(surv)}）:")
        for src, cnt in Counter(s["src"] for s in surv).most_common():
            print(f"  {src}: {cnt}")
        for s in sorted(surv, key=lambda x: x["dur"]):
            print(f"  {s['w']:14s} dur={s['dur']:.2f}s src={s['src']:16s} utt=..{s['utt'][-20:]}")


if __name__ == "__main__":
    main()
