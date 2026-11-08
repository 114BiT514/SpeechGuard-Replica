"""[M1 收尾] 自录集机器预审：音频质量 + 朗读准确率（人工审听的前置筛选）。

定位：README/M4 声明自录集"发音不标准、待人工审听，不进主评估"。本脚本在
人工审听之前先量化两件事，回答"到底能不能用"：
  1. 信号质量：峰值/削波事件/噪声底/语音-底噪比/静音占比/直流偏移；
  2. 朗读质量：wav2vec2 CTC 词级时间戳转写 → 与 docs/self_recorded_script.md
     的朗读文本（Part A-D 63 句）做 difflib 词级对齐 → 逐句/逐 Part WER、
     疑似跳读句、Part E（中文）区间定位。
校准锚点：同一 CTC 贪心管线在 LibriSpeech test-clean 全集 WER=11.28%
（experiments/results/table3_error_rates.json），志愿者 WER 高出部分是
"口音/底噪/朗读偏差"的合成效应，供逐条复听排序用，不替代人工结论。

运行：
  E:/Miniconda/envs/speechguard/python.exe scripts/self_recorded_quality_eval.py            # 第一阶段（CTC）
  E:/Miniconda/envs/speechguard/python.exe scripts/self_recorded_quality_eval.py --conformer # 追加 Part A 区 Conformer 交叉验证

输出：
  data/self_recorded/quality_eval.json    全部数字（句级明细）
  data/self_recorded/quality_report.md    人工复听点位清单（按人×按句）
  data/self_recorded/transcripts/ctc_times_{person}.json   词级时间戳缓存
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

SR = 16000
SELF = os.path.join(PROJ, "data", "self_recorded")
OUT16K = os.path.join(SELF, "processed", "full16k")
TRANS = os.path.join(SELF, "transcripts")

# ─────────────────────────── 朗读文本参照（docs/self_recorded_script.md） ───────────────────────────

REF = []  # (part, sid, text)


def _add(part, sids, texts):
    REF.extend((part, s, t) for s, t in zip(sids, texts))


_add("A", [f"A{i:02d}" for i in range(1, 31)], [
    "The weather today is warmer than yesterday.",
    "I take the bus to campus every morning.",
    "My favorite season is autumn because of the falling leaves.",
    "Coffee helps me stay awake during long study sessions.",
    "The library closes at ten o'clock tonight.",
    "We watched a documentary about the ocean last weekend.",
    "My roommate plays the guitar very well.",
    "I need to buy a new pair of running shoes.",
    "The train to the city leaves twice every hour.",
    "Learning a new language takes patience and practice.",
    "Our team won the basketball game by two points.",
    "I usually cook noodles when I get home late.",
    "The museum opens free of charge on Sundays.",
    "My little sister is learning to ride a bicycle.",
    "This notebook has survived three semesters of lectures.",
    "The mountains look beautiful covered in morning fog.",
    "I forgot my umbrella and got soaked on the way home.",
    "Reading before bed helps me sleep better.",
    "The new coffee shop next to the gate opens tomorrow.",
    "Our professor postponed the deadline until Friday.",
    "Birds start singing outside my window at dawn.",
    "I prefer texting over phone calls.",
    "The park is full of children on sunny afternoons.",
    "My laptop battery dies within two hours now.",
    "We are planning a hiking trip for the holiday.",
    "The soup needs a little more salt.",
    "I have never been to the western provinces.",
    "Street food near the station is cheap and tasty.",
    "My grandfather tells the same stories every holiday.",
    "A short walk after dinner clears my mind.",
])
_add("B", ["B1", "B2", "B3"], [
    "My typical day starts at seven in the morning. I grab a quick breakfast, "
    "then ride my bike to the lecture hall. Classes keep me busy until noon, and "
    "I usually eat lunch with my friends at the canteen. In the afternoon I "
    "prefer to study in the library, where the atmosphere is quiet and focused. "
    "After dinner, I either go for a jog around the lake or review my notes in "
    "the dormitory.",
    "Last summer I took a train to the coast with two of my classmates. We left "
    "early in the morning and arrived before noon. The seaside town was smaller "
    "than we expected, but the beaches were clean and quiet. We rented two "
    "bicycles and rode along the shore until sunset. On the last day, we visited "
    "an old lighthouse and took dozens of photos. It was a simple trip, but one "
    "of the most relaxing weeks I have ever had.",
    "I started building model airplanes when I was in primary school. My first "
    "model was clumsy, with crooked wings and too much glue. Over the years I "
    "learned to sand the frame, balance the wings, and paint thin, even coats. "
    "Now I build one model every few months, and each one sits on the shelf "
    "above my desk. What I enjoy most is not the finished plane, but the quiet "
    "hours spent on each small part.",
])
_add("C", [f"C{i:02d}" for i in range(1, 21)], [
    "My name is Daniel Reed, and I live at forty-two Orchard Lane, Springfield.",
    "You can reach me on my cell phone at five five five, zero one four two.",
    "I was born on June the third, nineteen ninety-eight, at City Hospital.",
    "My passport number is nine three two four one, and it expires next spring.",
    "The password for my email is my daughter's birthday and the word sunshine.",
    "Please transfer the rent to my bank card ending in seven seven one nine.",
    "Doctor Miller asked me to take one tablet of the blue medicine every morning.",
    "I have a cardiology appointment at the clinic on Maple Street this Thursday.",
    "My grandmother is diabetic, so we keep insulin in the refrigerator.",
    "I attend the community church on River Road every Sunday morning.",
    "As a union member, I vote in the factory elections each autumn.",
    "My colleague Ahmed and his family moved here from another country two years ago.",
    "I support the green party in local elections, though I keep it private.",
    "My home address is apartment twelve, ninety-five Birch Avenue, Brookside.",
    "The recovery code for my account is four seven two, eight eight, zero six.",
    "My ID card was issued last March and carries my biometric data.",
    "Miss Chen from human resources has my social insurance number on file.",
    "I visit a therapist downtown every other week for my anxiety.",
    "Our landlord, Mister Rossi, can be reached at six six six, zero zero two five.",
    "The clinic said my blood test results will be ready on Monday morning.",
])
_add("D", [f"D{i:02d}" for i in range(1, 11)], [
    "The meeting starts at nine fifteen and ends at noon.",
    "My flight number is C A one eight five seven, boarding at gate twenty-two.",
    "The total comes to one hundred forty-two dollars and sixty cents.",
    "Please enter the verification code: three, nine, one, seven, five.",
    "The package will arrive between two and four in the afternoon.",
    "Room six oh four is on the sixth floor, next to the elevator.",
    "The license plate reads B, seven, K, two, nine, one.",
    "We will celebrate my birthday on December the twelfth this year.",
    "The discount code is SAVE twenty, valid until October thirty-first.",
    "Dial one zero two four for the campus emergency line.",
])

WEAK = {"a", "i", "o"}  # 中文段英文 CTC 易幻听出的单字母词

# "读出小节标题/序号"的 off-script 词汇（朗读规范：标题不读出声）。
# 收紧规则：只有含 "part" 的插入组、或紧跟 part 锚点 ≤10s 的"节字母/数字"组才算
# 标题复读——否则句中真实词（如 twenty-nine 的 nine）对齐错位会被误剔。
TITLE_ANCHOR = "part"
TITLE_FOLLOW = {"a", "b", "c", "d", "e",
                "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve"}


def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z']+", s.lower())


# ─────────────────────────── 信号质量 ───────────────────────────

def signal_metrics(wav: np.ndarray) -> dict:
    frame = int(0.03 * SR)
    n = len(wav) // frame
    fr = wav[: n * frame].reshape(n, frame)
    rms_db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-12)
    clip = np.abs(wav) >= 0.999
    d = np.diff(clip.astype(np.int8))
    bursts = int((d == 1).sum()) + (1 if clip[0] else 0)
    nf, sp = float(np.percentile(rms_db, 10)), float(np.percentile(rms_db, 90))
    return {"dur": len(wav) / SR, "peak": float(np.abs(wav).max()),
            "clip_frac": float(clip.mean()), "clip_bursts": bursts,
            "noise_floor_db": round(nf, 1), "speech_db": round(sp, 1),
            "snr_proxy_db": round(sp - nf, 1),
            "silence_frac": float((rms_db < nf + 12).mean()),
            "dc_offset": float(np.mean(wav))}


# ─────────────────────────── CTC 词级时间戳（带缓存） ───────────────────────────

def chunk_bounds(wav):
    frame = int(0.03 * SR)
    n = len(wav) // frame
    rms = np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12)
    bounds, pos = [0], 0
    while (n - pos) * 0.03 > 30:
        lo = pos + int(22 / 0.03)
        hi = pos + int(30 / 0.03)
        cut = lo + int(np.argmin(rms[lo:hi]))
        bounds.append(cut)
        pos = cut
    bounds.append(n)
    return [(b * frame, e * frame) for b, e in zip(bounds[:-1], bounds[1:])]


def ctc_words(stem: str, wav: np.ndarray, device: str, fresh=False):
    cache = os.path.join(TRANS, f"ctc_times_{stem}.json")
    if os.path.exists(cache) and not fresh:
        return [tuple(w) for w in json.load(open(cache, encoding="utf-8"))]
    from src.content.transcribe import transcribe_with_times
    out = []
    for s, e in chunk_bounds(wav):
        for w, t0, t1 in transcribe_with_times(wav[s:e].astype(np.float32),
                                               device=device):
            out.append((w, t0 + s / SR, t1 + s / SR))
    os.makedirs(TRANS, exist_ok=True)
    json.dump(out, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    return out


# ─────────────────────────── 词级对齐 → 句级 WER ───────────────────────────

def align(ref_words, ref_sent, hyp):
    """difflib 词级对齐；插入归到其前面那个 ref 词所在的句。
    中文尾部可能被英文 CTC 幻听出孤立 a/i/o 并误配成"匹配"——把弱 token 匹配
    降级：仅当它出现在最后一个**实匹配块（size≥2 或非弱 token）**之后 3s 以外。"""
    hyp_norm = [re.sub(r"[^a-z']", "", w.lower()) for w, _, _ in hyp]
    sm = difflib.SequenceMatcher(None, ref_words, hyp_norm, autojunk=False)
    blocks = list(sm.get_opcodes())
    status = [None] * len(ref_words)      # (kind, hyp_idx)

    # pass 1：equal 赋值 + 最后一个"实匹配"时间；弱 token 单配则降级
    solid_t, weak_ones = 0.0, []
    for tag, i1, i2, j1, j2 in blocks:
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            status[i1 + k] = ("M", j1 + k)
        if i2 - i1 >= 2 or ref_words[i1] not in WEAK:
            solid_t = max(solid_t, hyp[j2 - 1][2])
        else:
            weak_ones.append((i1, j1))
    demoted = set()
    for i, j in weak_ones:
        if hyp[j][1] > solid_t + 3.0:     # 尾部中文幻听 → 撤销匹配
            status[i] = ("D", None)
            demoted.add(j)

    # pass 2：收尾 S/D 配对与插入分组（opcodes 互斥，replace 词此时必为 None）
    ins_groups = defaultdict(list)        # 边界 b -> [[hyp_idx...], ...]
    used = set()
    for tag, i1, i2, j1, j2 in blocks:
        if tag == "equal":
            for k in range(i2 - i1):
                if status[i1 + k][1] is not None:
                    used.add(j1 + k)
        elif tag == "replace":
            free = [j for j in range(j1, j2) if j not in demoted]
            for k in range(i2 - i1):
                if free:
                    status[i1 + k] = ("S", free.pop(0))
                    used.add(status[i1 + k][1])
                else:
                    status[i1 + k] = ("D", None)
            if free:
                ins_groups[i2].append(free)
        elif tag == "delete":
            for k in range(i2 - i1):
                status[i1 + k] = ("D", None)
        elif tag == "insert":
            ins_groups[i1].append(list(range(j1, j2)))
    for k in range(len(ref_words)):
        if status[k] is None:
            status[k] = ("D", None)

    sent = {}
    for i, s in enumerate(ref_sent):
        st = sent.setdefault(s, {"sid": s, "n": 0, "M": 0, "S": 0, "D": 0,
                                 "ins": 0, "t0": None, "t1": None,
                                 "hyp_words": []})
        st["n"] += 1
        kind, j = status[i]
        st[kind] += 1
        if j is not None:
            w, t0, t1 = hyp[j]
            st["t0"] = t0 if st["t0"] is None else min(st["t0"], t0)
            st["t1"] = t1 if st["t1"] is None else max(st["t1"], t1)
            st["hyp_words"].append((t0, w))
    # 插入组归属：含 "part" 的组为标题锚点；锚点后 ≤10s 内纯节字母/数字的组
    # 一并算 off-script 剔除。其余插入计入所在句 WER。
    events = []
    for b, groups in ins_groups.items():
        for g in groups:
            toks = [hyp_norm[j] for j in g]
            events.append((hyp[g[0]][1], b, g, toks))
    events.sort(key=lambda e: e[0])
    offscript, last_anchor = [], None
    for t, b, g, toks in events:
        is_title = (len(g) <= 4 and TITLE_ANCHOR in toks) or \
                   (last_anchor is not None and t - last_anchor <= 10.0
                    and len(g) <= 4 and all(x in TITLE_FOLLOW for x in toks))
        if is_title:
            offscript.extend(hyp[j] for j in g)
            if TITLE_ANCHOR in toks:
                last_anchor = t
            continue
        if 0 < b < len(ref_words):
            sent[ref_sent[b - 1]]["ins"] += len(g)

    matched_t = [hyp[j][1:3] for _, j in status if j is not None]
    first_used = min((a for a, _ in matched_t), default=0.0)
    last_used = max((b for _, b in matched_t), default=0.0)
    before = [(w, t0, t1) for w, t0, t1 in hyp if t1 <= first_used]
    tail = [(w, t0, t1) for w, t0, t1 in hyp if t0 >= last_used]

    for st in sent.values():
        st["wer"] = (st["S"] + st["D"] + st["ins"]) / st["n"] if st["n"] else 0.0
        st["skipped"] = bool(st["M"] == 0 and st["S"] == 0)
        st["hyp_words"].sort()
        st["hyp_text"] = " ".join(w for _, w in st["hyp_words"])[:120]
    return sent, {"first_used": first_used, "last_used": last_used,
                  "before": before, "tail": tail, "offscript": offscript}


# ─────────────────────────── 每人评估 ───────────────────────────

def person_part_wers(sent):
    out = {}
    for part in "ABCD":
        sids = [s for p, s, _ in REF if p == part]
        n = sum(sent[s]["n"] for s in sids)
        e = sum(sent[s]["S"] + sent[s]["D"] + sent[s]["ins"] for s in sids)
        out[part] = round(100 * e / n, 1)
    n = sum(v["n"] for v in sent.values())
    e = sum(v["S"] + v["D"] + v["ins"] for v in sent.values())
    out["ALL"] = round(100 * e / n, 1)
    return out


def evaluate(stem, device, fresh=False):
    wav_path = os.path.join(OUT16K, f"{stem}.wav")
    wav, sr = sf.read(wav_path)
    assert sr == SR, (wav_path, sr)
    ref_words, ref_sent = [], []
    for _p, sid, text in REF:
        for w in tokenize(text):
            ref_words.append(w)
            ref_sent.append(sid)
    hyp = ctc_words(stem, wav, device, fresh=fresh)
    sent, region = align(ref_words, ref_sent, hyp)
    return {"person": stem, "signal": signal_metrics(wav),
            "ctc_words": len(hyp),
            "part_wer": person_part_wers(sent),
            "skipped_sids": [s["sid"] for s in sent.values() if s["skipped"]],
            "regions": {"before_words": len(region["before"]),
                        "tail_start": round(region["tail"][0][1], 1) if region["tail"] else None,
                        "tail_end": round(region["tail"][-1][2], 1) if region["tail"] else None,
                        "tail_words": len(region["tail"]),
                        "offscript": [f"{w}@{t0:.0f}s" for w, t0, _ in region["offscript"]]},
            "sentences": {s: {k: v for k, v in st.items() if k != "hyp_words"}
                          for s, st in sent.items()},
            "_wav": wav}


def verdict(r):
    """筛查级判定（非最终结论，人工复听为准）。warns 影响初判，infos 仅提示。"""
    sig, wer = r["signal"], r["part_wer"]
    warns, infos = [], []
    if sig["clip_bursts"] > 5:
        warns.append(f"削波事件 {sig['clip_bursts']} 次（录入电平过高）")
    if sig["noise_floor_db"] > -30:
        warns.append(f"底噪偏高 {sig['noise_floor_db']}dBFS")
    if sig["snr_proxy_db"] < 10:
        warns.append(f"语音-底噪比低 {sig['snr_proxy_db']}dB")
    if wer["ALL"] > 30:
        warns.append("CTC-WER 远高于 LibriSpeech 锚点 11.3%")
    if len(r["skipped_sids"]) > 3:
        warns.append(f"疑似跳读 {len(r['skipped_sids'])} 句")
    if not r["regions"]["tail_start"]:
        infos.append("未检测到 Part E（可选项，正常）")
    level = ("可用候选" if wer["ALL"] <= 20 and not warns
             else "边缘，需重点复听" if wer["ALL"] <= 35 and not warns
             else "需重点复听" if warns or wer["ALL"] <= 45 else "发音问题显著")
    return level, warns, infos


# ─────────────────────────── 报告 ───────────────────────────

def fmt_t(s):
    if s is None:
        return "--:--"
    return f"{int(s) // 60:02d}:{int(s) % 60:02d}"


REF_TEXT = {sid: t for _p, sid, t in REF}


def write_report(results, path):
    lines = ["# 自录集机器预审报告（人工复听前）",
             "",
             "> 方法：wav2vec2 CTC 词级时间戳 → 与朗读文本 A-D（63 句）difflib 词级对齐 → 句级 WER。",
             "> 锚点：同一 CTC 贪心管线在 LibriSpeech test-clean 全集 WER=11.28%"
             "（table3_error_rates.json）；志愿者高出部分≈口音+底噪+朗读偏差的合成，"
             "**最终以人工复听为准**。", "", "## 总览", "",
             "| 人 | 时长 | 峰值 | 削波 | 底噪 | SNR代 | 静音% | CTC-WER A/B/C/D/全 | 跳读 | PartE 区间 | 初判 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        sig, w = r["signal"], r["part_wer"]
        lv, _, _ = verdict(r)
        tr = r["regions"]
        lines.append(f"| {r['person']} | {sig['dur'] / 60:.1f}min | {sig['peak']:.2f} | "
                     f"{sig['clip_bursts']} | {sig['noise_floor_db']}dB | "
                     f"{sig['snr_proxy_db']}dB | {sig['silence_frac'] * 100:.0f}% | "
                     f"{w['A']}/{w['B']}/{w['C']}/{w['D']}/{w['ALL']} | "
                     f"{len(r['skipped_sids'])} | "
                     f"{fmt_t(tr['tail_start'])}-{fmt_t(tr['tail_end'])} | {lv} |")
    lines += ["", "## 每人复听点位", ""]
    for r in results:
        lines.append(f"### {r['person']}")
        lv, warns, infos = verdict(r)
        lines.append(f"- 初判：**{lv}**" +
                     ("" if not warns else "；" + "；".join(warns)) +
                     ("" if not infos else "（注：" + "；".join(infos) + "）"))
        cw = r.get("conformer_wer")
        if cw is not None:
            cw_txt = (", ".join(f"{k}={v}%" for k, v in cw.items())
                      if isinstance(cw, dict) else f"{cw}%")
            lines.append(f"- Conformer（论文评估模型口径，A/C/D 逐句）WER：{cw_txt}")
        sent = r["sentences"]
        worst = sorted((s for s in sent.values() if s["n"]), key=lambda s: -s["wer"])
        bad = [s for s in worst if s["wer"] >= 0.4][:12]
        if bad:
            lines.append("- 高错误句（WER≥40%，前 12 条）：")
            for s in bad:
                tag = "（疑似跳读）" if s["skipped"] else ""
                lines.append(f"  - [{fmt_t(s['t0'])}] {s['sid']} WER={s['wer'] * 100:.0f}%{tag} "
                             f"应有=\"{REF_TEXT[s['sid']][:60]}\" 听到=\"{s['hyp_text'][:55]}\"")
        if r["skipped_sids"]:
            lines.append(f"- 疑似跳读句：{', '.join(r['skipped_sids'])}")
        tr = r["regions"]
        if tr["tail_start"]:
            lines.append(f"- 尾部未对齐区（疑 Part E 中文）：{fmt_t(tr['tail_start'])}-"
                         f"{fmt_t(tr['tail_end'])}，{tr['tail_words']} 词")
        if tr["before_words"]:
            lines.append(f"- 开头未对齐：{tr['before_words']} 词（可能是读出'PART A'标题等）")
        if tr["offscript"]:
            lines.append(f"- 读出标题/序号（off-script，已从 WER 剔除）：{', '.join(tr['offscript'])}")
        lines.append("")
    open(path, "w", encoding="utf-8").write("\n".join(lines))


# ─────────────────────────── Conformer 交叉验证（第二阶段） ───────────────────────────


def _silence_gaps(wav: np.ndarray, nf_db: float, min_len: float = 0.2):
    """波形 → 静音区间 [(s, e)]（秒）：10ms 帧能量 < max(底噪+10dB, P95-25dB)。"""
    frame = int(0.01 * SR)
    n = len(wav) // frame
    if n == 0:
        return []
    db = 20 * np.log10(np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(1)) + 1e-12)
    thr = max(nf_db + 10.0, float(np.percentile(db, 95)) - 25.0)
    silent = db < thr
    gaps, start = [], None
    for i, s in enumerate(silent):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if (i - start) * 0.01 >= min_len:
                gaps.append((start * 0.01, i * 0.01))
            start = None
    if start is not None and (n - start) * 0.01 >= min_len:
        gaps.append((start * 0.01, n * 0.01))
    return gaps


def _boundary(t_prev_end: float, t_next_start: float, gaps) -> float:
    """句界：上句尾与下句头的中点，吸附到包含它的静音段中心（否则取 ±1s 内
    最近段中心；再不行用中点）。"""
    mid = (t_prev_end + t_next_start) / 2
    best, bestd = None, 1.0
    for gs, ge in gaps:
        if gs <= mid <= ge:
            return (gs + ge) / 2
        if ge - gs >= 0.25:
            d = abs((gs + ge) / 2 - mid)
            if d < bestd:
                best, bestd = (gs + ge) / 2, d
    return best if best is not None else mid


def _sentence_windows(sent, wav, nf_db):
    """给每句（REF 顺序，含 B）算切分窗口 [lo, hi]。
    边界 = 上句 t1 与下句 t0 的中点吸附静音段——录音规范句间停 1-2s。
    注意不能用相邻句 t0 的中点：句长 4-6s 而停顿仅 1-2s，t0 中点会落进上句
    语音里，把上句句尾泄漏进本句窗口（person4 对照实验暴露：Conformer 把
    泄漏尾音连读出来，WER 虚高一倍）。切出的每句完整且分布内（Conformer 是
    6-8s 单句训练的 Encoder-Decoder，±0.4s 硬切会切掉句首句尾诱发脱轨幻听，
    30s 长块则慢一个数量级）。句间标题复读（"PART B"）落在窗外天然排除。"""
    have = [(p, s, sent[s]["t0"], sent[s]["t1"]) for p, s, _t in REF
            if sent[s]["t0"] is not None]
    gaps = _silence_gaps(wav, nf_db)
    wins = {}
    for i, (part, sid, t0, t1) in enumerate(have):
        t1 = t1 if t1 is not None else t0 + 3.0
        if i > 0:
            pt1 = have[i - 1][3] if have[i - 1][3] is not None else have[i - 1][2]
            lo = _boundary(pt1, t0, gaps)
        else:
            lo = t0 - 0.5
        if i + 1 < len(have):
            hi = _boundary(t1, have[i + 1][2], gaps)
        else:
            # 段末句无上句可借：吸附到 t1 之后第一段静音中心（否则 t1 被 CTC
            # 尾部拉伸时会把段后朗读吞进窗口）
            hi = t1 + 1.5
            for gs, ge in gaps:
                if gs >= t1 - 0.2:
                    hi = (gs + ge) / 2
                    break
        lo = max(lo, t0 - 2.5)                       # 防上句 t1 拉伸过头吞并本句
        cap = 60.0 if part == "B" else 12.0          # 防 t1 拉伸把下句圈进来
        hi = min(max(hi, t1), t0 + cap)
        if hi <= lo + 0.5:
            hi = lo + 0.5
        wins[sid] = (lo, hi)
    return wins


def conformer_pass(results, device):
    """A/C/D 逐句（窗口见 _sentence_windows）→ Conformer（beam=4+LM，与主表
    WER 同配置）→ 按 Part 聚合 WER。回答两件事：① 带 LM 的论文评估模型能否
    听懂（分离"发音差"vs"CTC 贪心放大"）；② 未来 Volunteers 列 Ori 态
    Conformer WER 的量级。B 段连续朗读不切、跳过（CTC 已覆盖）。"""
    import jiwer
    import time
    import torch
    sys.path.insert(0, os.path.join(PROJ, "experiments"))
    from eval_asr import load_asr, transcribe, normalize

    ref_text = {sid: t for _p, sid, t in REF}
    model = load_asr(device=device)
    for r in results:
        sent = r["sentences"]
        wins = _sentence_windows(sent, r["_wav"], r["signal"]["noise_floor_db"])
        clips, sids = [], []
        for p, sid, _t in REF:
            if p == "B" or sid not in wins:
                continue
            lo, hi = wins[sid]
            clips.append(r["_wav"][max(int(lo * SR), 0):
                                    int(hi * SR)].astype(np.float32))
            sids.append((p, sid))
        t0 = time.time()
        hyps = []
        for i in range(0, len(clips), 20):            # 分批打印进度
            hyps += transcribe(model, clips[i:i + 20], batch_size=2)
            print(f"  {r['person']}: {min(i + 20, len(clips))}/{len(clips)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        hyp_of = dict(zip(sids, hyps))
        part = defaultdict(lambda: [0, 0])
        for p, sid, _t in REF:
            if p == "B":
                continue
            st = sent[sid]
            n = st["n"]
            part[p][1] += n
            if sid not in wins:                       # 跳读句：全算错
                part[p][0] += n
                continue
            w = jiwer.wer(normalize(ref_text[sid]), normalize(hyp_of[(p, sid)]))
            part[p][0] += round(w * n)
        r["conformer_wer"] = {p: round(100 * e / n, 2) for p, (e, n) in sorted(part.items())}
        print(f"{r['person']}: conformer WER (A/C/D) = {r['conformer_wer']}", flush=True)
        del clips, hyps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────── main ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fresh", action="store_true", help="忽略词级时间戳缓存重转写")
    ap.add_argument("--conformer", action="store_true",
                    help="追加：A/C/D 逐句切段跑 Conformer（论文评估模型）交叉验证")
    ap.add_argument("--persons", default=None,
                    help="逗号分隔的人名过滤（如 person3,person8），默认全部")
    args = ap.parse_args()

    people = sorted(f[:-4] for f in os.listdir(OUT16K) if f.endswith(".wav"))
    if args.persons:
        want = set(args.persons.split(","))
        people = [p for p in people if p in want]
    results = []
    for p in people:
        print(f"eval {p} ...", flush=True)
        results.append(evaluate(p, args.device, fresh=args.fresh))

    out_json = os.path.join(SELF, "quality_eval.json")
    old = {}
    if os.path.exists(out_json):
        old = {r["person"]: r for r in json.load(open(out_json, encoding="utf-8"))}
    if args.conformer:
        todo = [r for r in results
                if args.fresh or "conformer_wer" not in old.get(r["person"], {})]
        if len(todo) < len(results):
            print(f"skip {len(results) - len(todo)} persons with existing "
                  f"conformer results (use --fresh to force)", flush=True)
        conformer_pass(todo, args.device)
    # 本轮没跑的人：沿用既有 Conformer 数字
    for r in results:
        if "conformer_wer" not in r and "conformer_wer" in old.get(r["person"], {}):
            r["conformer_wer"] = old[r["person"]]["conformer_wer"]
    for r in results:
        old[r["person"]] = {k: v for k, v in r.items() if k != "_wav"}
    merged = [old[k] for k in sorted(old)]
    json.dump(merged, open(out_json, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False, default=str)
    write_report(merged, os.path.join(SELF, "quality_report.md"))
    print("\nperson | 初判 | CTC-WER A/B/C/D/ALL | conformer A/C/D")
    for r in merged:
        w = r["part_wer"]
        print(f"{r['person']} | {verdict(r)[0]} | {w['A']}/{w['B']}/{w['C']}/"
              f"{w['D']}/{w['ALL']} | {r.get('conformer_wer')}")
    print("\n->", out_json)


if __name__ == "__main__":
    main()
