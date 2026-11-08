"""[M8 补空] Volunteers 列全套实验：Table 5 行 + Table 7 FNR + RTC + 三层分组分析。

素材：data/self_recorded/processed/utterances/（scripts/self_recorded_segment.py 产出，
8 人 × A/C/D 60 短句 = 480 条，58.6 分钟；B 段落不进主协议——论文 volunteers 粒度
为 ~8s utterance）。

协议对齐（与 LS/VCTK 主表同一套代码路径，保证行间可比）：
  warping  single 模式、n∈[1,10]、dist∈(0.5,0.6)、逐句 rng 流（seed=505，逐句种子
           为 main_table 权威口径）；att= reducing 攻击（攻击者已知 n/dist 范围，
           随机参数逆扭曲，exp3 同款）
  EER      ECAPA-TDNN 全对配对：genuine=同人异句、impostor=跨人（480 句全量）
  WER      Conformer beam=4；每人抽 30 句分层子集（seed=505）
  FNR      Part C+D 全部 240 句：utterance 级 CTC 词时间戳（检测-时间戳同源）
           → 词表+模式+NER 检测 → AES-CTR 帧加密(margin=8) → Conformer → 隐藏率；
           Dec 态逐字节还原断言（论文 Table 7 Dec=Ori 口径）
  RTC      protect/recover 每音频秒耗时（中位数）

分组（我们独有，论文没有的分层证据）：
  native（person8）/ clear（person1/2/3/4/5/7）/ colloquial（person6）

运行（GPU ~40-55 分钟，ASR 磁盘缓存可断点续跑）：
  E:/Miniconda/envs/speechguard/python.exe experiments/exp12_volunteers.py
输出：experiments/results/volunteers_main.json + data/probe/volunteers/ 试听样例
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from experiments.eval_asr import normalize  # noqa: E402

RES = os.path.join(PROJ, "experiments", "results")
UTT_DIR = os.path.join(PROJ, "data", "self_recorded", "processed", "utterances")
MANIFEST = os.path.join(PROJ, "data", "self_recorded", "processed", "volunteers_manifest.json")
HYP_CACHE = os.path.join(RES, "volunteers_hyp_cache.json")
PROBE = os.path.join(PROJ, "data", "probe", "volunteers")

SR = 16000
SEED = 505
DEVICE = "cuda:0"
MARGIN = 8
WER_PER_PERSON = 30


def _attack_single(prot_wav, rng):
    """reducing 攻击（exp3 同款）：随机参数逆扭曲保护谱。"""
    from src.acoustic.warping import SegmentParams, _apply_single, gen_params
    a, b, d = gen_params(rng=rng)
    fake = [SegmentParams(0, max(prot_wav.size // 576, 1), a, b, d)]
    out, _ = _apply_single(prot_wav, archive=fake, invert=True)
    return out


def main():
    import jiwer
    from experiments.eval_asv import eer_from_scores, embed_wavs, load_asv
    from experiments.eval_asr import load_asr, transcribe
    from src.acoustic.warping import protect, recover
    from src.content.annotation import Annotation
    from src.content.detect import detect_sensitive
    from src.content.encrypt import decrypt_spans, encrypt_spans, gen_key
    from src.content.sensitive_lists import SensitiveLists
    from src.content.transcribe import transcribe_with_times

    rng = np.random.default_rng(SEED)
    rows = [r for r in json.load(open(MANIFEST, encoding="utf-8"))
            if r["part"] in "ACD"]
    # 数字序排序：person10/11 不得按字典序插到 person2 前面——否则 protect 的
    # rng 流位置整体漂移，旧 8 人的三态参数与 ASR 缓存全部作废重跑
    rows.sort(key=lambda r: (int(r["person"][6:]), r["sid"]))
    print(f"utterances: {len(rows)} 条 / "
          f"{sum(r['dur_s'] for r in rows) / 60:.1f} min / "
          f"{len({r['person'] for r in rows})} 人", flush=True)

    # ── 三态音频（warping 路径）──
    print("== protect / attack / recover ==", flush=True)
    t0 = time.time()
    rtc_p, rtc_r = [], []
    for r in rows:
        wav, _ = sf.read(os.path.join(UTT_DIR, r["person"], f"{r['sid']}.wav"))
        r["wav"] = np.asarray(wav, np.float64)
        t1 = time.time()
        prot, arch = protect(r["wav"], rng=rng)
        r["prot"], r["arch"] = prot, arch
        r["att"] = _attack_single(prot, rng)
        rtc_p.append((time.time() - t1) / max(len(wav) / SR, 1e-6))
        t1 = time.time()
        r["rec"] = recover(prot, arch)
        rtc_r.append((time.time() - t1) / max(len(wav) / SR, 1e-6))
        r["pcm"] = (np.clip(r["wav"], -1, 1) * 32767).astype(np.int16)
    print(f"  {len(rows)} 句三态生成，{time.time() - t0:.0f}s；"
          f"RTC protect={np.median(rtc_p):.3f} recover={np.median(rtc_r):.3f}",
          flush=True)
    os.makedirs(PROBE, exist_ok=True)
    for r in rows[:3] + rows[240:242]:
        for st in ("wav", "prot", "att", "rec"):
            sf.write(os.path.join(PROBE, f"{r['utt']}_{st}.wav"),
                     np.asarray(r[st], np.float32), SR)

    # ── EER（全量 480，全对配对）──
    print("== EER（ECAPA 全对配对）==", flush=True)
    asv = load_asv(device=DEVICE)
    spk = np.array([r["person"] for r in rows])
    embs = {st: embed_wavs(asv, [np.asarray(r[st], np.float32) for r in rows])
            for st in ("wav", "prot", "att", "rec")}
    n = len(rows)
    same = [np.where(spk == spk[i])[0] for i in range(n)]
    def pairs(state, diff):
        sc = []
        for i in range(n):
            idx = np.where(spk != spk[i])[0] if diff else same[i][same[i] != i]
            sc.append((embs[state][i] * embs[state][idx]).sum(1))
        return np.concatenate(sc)
    eer = {st: round(100 * eer_from_scores(pairs(st, False), pairs(st, True)), 2)
           for st in ("wav", "prot", "att", "rec")}
    print("EER Ori/Pro/Att/Rec:", eer, flush=True)

    # ── WER（每人抽 30 句分层子集）──
    print("== WER（Conformer beam=4）==", flush=True)
    sub = []
    plist = sorted({r["person"] for r in rows}, key=lambda p: int(p[6:]))
    for pi, p in enumerate(plist):
        cand = [r for r in rows if r["person"] == p]
        pick = np.random.default_rng(SEED + 1 + 100 * pi).choice(   # 逐人独立种子
            len(cand), size=min(WER_PER_PERSON, len(cand)), replace=False)
        sub += [cand[i] for i in sorted(pick)]
    cache = json.load(open(HYP_CACHE, encoding="utf-8")) if os.path.exists(HYP_CACHE) else {}
    model = load_asr(device=DEVICE, overrides="test_beam_size: 4")

    def asr_batch(items, key_fn, wav_fn):
        todo = [(key_fn(r), wav_fn(r)) for r in items if key_fn(r) not in cache]
        print(f"  缓存命中 {len(items) - len(todo)}/{len(items)}，待转 {len(todo)}", flush=True)
        t0 = time.time()
        done = 0
        for s in range(0, len(todo), 30):
            ch = todo[s:s + 30]
            for (k, _), h in zip(ch, transcribe(model, [w for _, w in ch], batch_size=2)):
                cache[k] = h
            done += len(ch)
            el = time.time() - t0
            print(f"  {done}/{len(todo)}（{el:.0f}s，"
                  f"剩 ~{el / done * (len(todo) - done):.0f}s）", flush=True)
            json.dump(cache, open(HYP_CACHE, "w", encoding="utf-8"), ensure_ascii=False)

    def corpus_wer(items, state):
        errs = tot = 0
        per = {}
        for r in items:
            h = normalize(cache[f"{state}|{r['utt']}"])
            ref = normalize(r["ref"])
            e = round(jiwer.wer(ref, h) * len(ref.split()))
            errs += e
            tot += len(ref.split())
            g = per.setdefault(r["group"], [0, 0])
            g[0] += e
            g[1] += len(ref.split())
        overall = round(100 * errs / tot, 2)
        groups = {g: round(100 * e / t, 2) for g, (e, t) in per.items()}
        return overall, groups

    wer_res = {}
    for state, wf in (("ori", lambda r: r["wav"]), ("pro", lambda r: r["prot"]),
                      ("att", lambda r: r["att"]), ("rec", lambda r: r["rec"])):
        asr_batch(sub, lambda r, s=state: f"W|{s}|{r['utt']}",
                  lambda r, f=wf: np.asarray(f(r), np.float32))
        wer_res[state] = corpus_wer(sub, f"W|{state}")
        print(f"  WER {state}: {wer_res[state][0]}%  分组 {wer_res[state][1]}", flush=True)

    # ── FNR（Part C+D 全部 240 句）──
    print("== FNR（Table 7 口径）==", flush=True)
    sl = SensitiveLists.build()
    cd = [r for r in rows if r["part"] in "CD"]
    n_words = 0
    for r in cd:
        ctc = transcribe_with_times(r["wav"].astype(np.float32), device=DEVICE)
        hits = detect_sensitive(ctc, sl, device=DEVICE)
        ann = Annotation.from_ctc(r["utt"], ctc)
        by_idx = {h.idx: h for h in hits}
        idxs = sorted(h.idx for h in hits)
        groups, cur = [], (idxs[:1] or [])
        for j in idxs[1:]:
            if j == cur[-1] + 1:
                cur.append(j)
            else:
                groups.append(cur)
                cur = [j]
        if cur:
            groups.append(cur)
        for g in groups:
            ann.mark_words(g, by_idx[g[0]].sources[0])
        spans = ann.merge_spans(margin_frames=MARGIN)
        keys = [gen_key() for _ in spans]
        r["spans"], r["keys"] = spans, keys
        r["enc_pcm"] = encrypt_spans(r["pcm"], spans, keys)
        assert np.array_equal(decrypt_spans(r["enc_pcm"], spans, keys), r["pcm"])
        r["words"] = sorted({h.word.upper().strip(".,!?;:") for h in hits})
        n_words += len(r["words"])
        r["cov"] = sum(fe - fs + 1 for fs, fe in spans) * 576 / max(len(r["pcm"]), 1)
    print(f"  检测：{len(cd)} 句 / {n_words} 敏感词，"
          f"平均加密覆盖 {100 * np.mean([r['cov'] for r in cd]):.1f}%", flush=True)
    asr_batch(cd, lambda r: f"ori|{r['utt']}", lambda r: r["wav"].astype(np.float32))
    asr_batch(cd, lambda r: f"enc|{r['utt']}",
              lambda r: r["enc_pcm"].astype(np.float32) / 32768.0)

    def fnr(items):
        tot = rec = 0
        per = {}
        for r in items:
            hyp = set(normalize(cache[f"enc|{r['utt']}"]).split())
            ori_ok = sum(1 for w in r["words"] if w in normalize(cache[f"ori|{r['utt']}"]).split())
            hit = sum(1 for w in r["words"] if w in hyp)
            tot += len(r["words"])
            rec += hit
            g = per.setdefault(r["group"], [0, 0, 0, 0])   # tot, rec, ori_tot, ori_rec
            g[0] += len(r["words"]); g[1] += hit
            g[2] += len(r["words"]); g[3] += ori_ok
        return (round(100 * (1 - rec / tot), 2) if tot else float("nan"),
                {g: round(100 * (1 - b / a), 2) for g, (a, b, _, _) in per.items()},
                tot, rec,
                {g: round(100 * (1 - d / c), 2) for g, (_, _, c, d) in per.items()})

    fnr_all, fnr_groups, tot, rec, ori_groups = fnr([r for r in cd if r["words"]])
    print(f"FNR Enc={fnr_all}%（{tot - rec}/{tot} 隐藏；分组 {fnr_groups}）", flush=True)
    print(f"Ori 基线（同分母）分组 {ori_groups}", flush=True)

    out = {
        "protocol": {"dataset": "self_recorded volunteers 8人", "seed": SEED,
                     "n_utts_eer": n, "n_utts_wer": len(sub), "n_utts_fnr": len(cd),
                     "warping": "single dist(0.5,0.6) n(1,10)", "margin_frames": MARGIN,
                     "asr": "Conformer beam=4", "asv": "ECAPA 全对配对",
                     "groups": {"native": "person8", "colloquial": "person6",
                                "clear": "person1/2/3/4/5/7"}},
        "table5_row": {"eer_ori": eer["wav"], "eer_pro": eer["prot"],
                       "eer_att": eer["att"], "eer_rec": eer["rec"],
                       "wer_ori": wer_res["ori"][0], "wer_pro": wer_res["pro"][0],
                       "wer_att": wer_res["att"][0], "wer_rec": wer_res["rec"][0],
                       "wer_by_group": {st: wer_res[st][1] for st in wer_res}},
        "table7_row": {"fnr_enc": fnr_all, "n_words": tot, "survived": rec,
                       "fnr_by_group": fnr_groups, "ori_by_group": ori_groups,
                       "mean_encrypt_coverage": round(
                           float(np.mean([r["cov"] for r in cd if r["words"]])), 4)},
        "rtc": {"protect": round(float(np.median(rtc_p)), 4),
                "recover": round(float(np.median(rtc_r)), 4)},
    }
    json.dump(out, open(os.path.join(RES, "volunteers_main.json"), "w",
                        encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(out["table5_row"], ensure_ascii=False, indent=1), flush=True)
    print("-> volunteers_main.json；试听样例 ->", PROBE, flush=True)


if __name__ == "__main__":
    main()
