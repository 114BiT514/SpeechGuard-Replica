"""[M4 任务1-4 整合] 敏感项自动检测流水线（论文 §4.3.1）。

输入是 CTC 词流（transcribe_with_times 的 [(word, t0, t1)]，词与时间戳同源，
不需要再对参考文本做 difflib 对齐——protect 阶段没有参考文本可用）。三类命中：

1. 词表命中（SensitiveLists.match_word，GDPR 类 + 自定义词）；
2. 模式命中（电话号码/日期/地址等正则，字符区间覆盖的词全部计入）；
3. NER 命中（人名/地点/机构/杂项实体，字符区间覆盖的词全部计入）。

虚词过滤（FNR 诊断的产物）：模式/NER 命中若落在英语封闭类虚词上（THE/AND/MAY/US…），
判为误命中剔除——LibriSpeech 里 NER 把 "TELLS US" 标成 LOC、date 正则把
"MAY VOYAGING" 整段吞掉都是这类噪声；真正的 GDPR 词表词不受影响。
国家名等真敏感内容通常以多词实体出现（UNITED STATES），仍会被保留。

输出按词序号去重；漏检由手动标注（annotation.py）增补——论文允许 4–10% 的
转写错误由人工修正。
"""
from __future__ import annotations

from dataclasses import dataclass

from src.content.sensitive_lists import SensitiveLists

# 英语封闭类虚词（检测误命中的主要来源）
FUNCTION_WORDS = {
    "A", "AN", "THE", "AND", "OR", "NOR", "BUT", "SO", "TO", "OF", "IN", "ON",
    "AT", "BY", "FOR", "WITH", "FROM", "AS", "IF", "THAN", "THEN",
    "BE", "BEEN", "BEING", "IS", "ARE", "WAS", "WERE", "AM", "DO", "DOES", "DID",
    "HAVE", "HAS", "HAD", "WILL", "WOULD", "SHALL", "SHOULD", "CAN", "COULD",
    "MIGHT", "MAY", "IT", "ITS", "THIS", "THAT", "THESE", "THOSE",
    "HE", "HIM", "HIS", "SHE", "HER", "HERS", "THEY", "THEM", "THEIR", "THEIRS",
    "WE", "US", "OUR", "OURS", "YOU", "YOUR", "YOURS", "I", "ME", "MY", "MINE",
    "NOT", "NO", "YES", "ALL", "ANY", "SOME", "EACH", "OTHER", "ANOTHER",
    "WHAT", "WHEN", "WHERE", "WHICH", "WHO", "WHOM", "WHOSE", "WHY", "HOW",
    "WHILE", "WITHIN", "UPON",
}


def is_function_word(w: str) -> bool:
    return w.upper().strip(".,!?;:\"'") in FUNCTION_WORDS


@dataclass
class Hit:
    idx: int          # 词序号（ctc_words 下标）
    word: str         # 词文本
    t0: float         # 起止时间（秒）
    t1: float
    sources: tuple[str, ...]   # ("vocab:health", "ner:PER", ...)


def detect_sensitive(ctc_words: list[tuple[str, float, float]],
                     sl: SensitiveLists | None = None,
                     device: str = "cuda:0", use_ner: bool = True,
                     use_patterns: bool = True,
                     filter_function_words: bool = True) -> list[Hit]:
    """词流 → 敏感词命中列表（按 idx 升序）。"""
    sl = sl or SensitiveLists.build()
    n = len(ctc_words)
    hits: dict[int, list[str]] = {}

    # 词流拼文本，记录每词字符区间（单词间单空格）
    parts, offs = [], []
    pos = 0
    for w, _, _ in ctc_words:
        parts.append(w)
        offs.append((pos, pos + len(w)))
        pos += len(w) + 1
    text = " ".join(parts)

    def cover(a: int, b: int, tag: str) -> None:
        for i, (s0, e0) in enumerate(offs):
            if s0 < b and e0 > a:
                hits.setdefault(i, []).append(tag)

    # 1. 词表
    for i, (w, _, _) in enumerate(ctc_words):
        cat = sl.match_word(w)
        if cat:
            hits.setdefault(i, []).append(f"vocab:{cat}")

    # 2. 模式（电话/日期/地址/编号）
    if use_patterns:
        for a, b, name in sl.match_pattern(text):
            cover(a, b, f"pattern:{name}")

    # 3. NER
    if use_ner and text.strip():
        from src.content.ner import extract_entities
        for _ent, lab, a, b in extract_entities(text, device=device):
            cover(a, b, f"ner:{lab}")

    out = []
    for i, srcs in sorted(hits.items()):
        if filter_function_words and is_function_word(ctc_words[i][0]):
            srcs = [s for s in srcs if s.startswith("vocab:")]  # 虚词只认词表命中
        if srcs:
            out.append(Hit(i, ctc_words[i][0], ctc_words[i][1], ctc_words[i][2],
                           tuple(sorted(set(srcs)))))
    return out
