"""[M4 任务4] 敏感词表管理（论文 §4.3.1）。

三类来源：
1. 预定义词表：按 GDPR 特殊类别（种族/政治/宗教/工会/基因/生物识别/健康/性取向）
   整理的英文信号词；
2. 模式（patterns）：电话号码、地址、日期等正则——论文用 NER 处理，本模块提供
   正则兜底（数字串、日期），NER 见 ner.py；
3. 自定义词表：音频主人自备（论文允许）。

词表在录音前本地存储（论文：hash tables，不对外共享），查找用 frozenset O(1)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# GDPR 特殊类别词表（§4.3.1 列举的类别，常见信号词；可扩展）
GDPR_WORDS: dict[str, set[str]] = {
    "health": {"DIABETIC", "DIABETES", "INSULIN", "THERAPIST", "THERAPY", "ANXIETY",
               "DEPRESSION", "PSYCHIATRIST", "HOSPITAL", "SURGERY", "PRESCRIPTION",
               "MEDICATION", "CARDIOLOGY", "BLOOD", "CANCER", "ASTHMA", "CLINIC"},
    "religion": {"CHURCH", "MOSQUE", "TEMPLE", "SYNAGOGUE", "PRIEST", "PASTOR",
                 "IMAM", "RABBI", "ATHEIST", "BAPTIZED", "RAMADAN",
                 "RELIGION", "RELIGIOUS", "FAITH", "MASS", "PRAYER"},
    "politics": {"ELECTION", "ELECTIONS", "VOTE", "VOTED", "VOTING", "DEMOCRAT",
                 "REPUBLICAN", "SOCIALIST", "COMMUNIST", "LIBERAL",
                 "CONSERVATIVE", "POLITICAL", "POLITICS", "UNION",
                 "STRIKE", "PROTEST", "DEMONSTRATION", "SENATOR", "PRESIDENT"},
    "origin": {"RACE", "RACIAL", "ETHNIC", "ETHNICITY", "IMMIGRANT", "REFUGEE",
               "ASYLUM", "NATIONALITY", "ANCESTRY", "TRIBE"},
    "biometric_genetic": {"BIOMETRIC", "FINGERPRINT", "DNA", "GENETIC", "GENOME",
                          "IRIS", "PASSPORT", "INSURANCE", "SOCIAL"},
    "private_life": {"SEXUAL", "ORIENTATION", "GAY", "LESBIAN", "MARRIED", "DIVORCED",
                     "PREGNANT", "ABORTION", "CONTRACEPTION"},
}

# 电话号码/证件号/日期等模式（转写文本多为大写单词或数字）
DEFAULT_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("phone_digits", re.compile(r"\b(?:\d[\s-]?){7,11}\d\b")),
    ("phone_words", re.compile(
        r"(?:\b(?:ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE)\b[ ,]?){7,12}")),
    ("date", re.compile(
        r"\b(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|"
        r"NOVEMBER|DECEMBER)\b(?:\s+\w+){0,3}")),
    ("address", re.compile(
        r"\b(?:APT|APARTMENT|STREET|ROAD|AVENUE|LANE|BOULEVARD|DRIVE)\b")),
    ("id_number", re.compile(r"\b[A-Z]{0,2}\d{6,}\b")),
]


@dataclass
class SensitiveLists:
    words: frozenset[str] = field(default_factory=frozenset)
    word_category: dict[str, str] = field(default_factory=dict)
    patterns: list[tuple[str, "re.Pattern[str]"]] = field(default_factory=list)

    @classmethod
    def build(cls, include_gdpr=True, custom_words=None, patterns=None,
              extra_patterns=None) -> "SensitiveLists":
        words: set[str] = set()
        cat: dict[str, str] = {}
        if include_gdpr:
            for c, ws in GDPR_WORDS.items():
                for w in ws:
                    words.add(w.upper())
                    cat.setdefault(w.upper(), c)
        for w in (custom_words or []):
            words.add(w.upper())
            cat.setdefault(w.upper(), "custom")
        pats = list(patterns if patterns is not None else DEFAULT_PATTERNS)
        pats += list(extra_patterns or [])
        return cls(frozenset(words), cat, pats)

    def match_word(self, word: str) -> str | None:
        """单词命中词表 → 返回类别，否则 None。"""
        w = word.upper().strip(".,!?;:\"'")
        return self.word_category.get(w)

    def match_pattern(self, text: str) -> list[tuple[int, int, str]]:
        """在转写文本上匹配模式 → [(start_char, end_char, category)]。"""
        hits = []
        for name, pat in self.patterns:
            for m in pat.finditer(text):
                hits.append((m.start(), m.end(), name))
        return sorted(hits)
