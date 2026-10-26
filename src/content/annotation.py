"""[M4 任务7] 手动标注数据结构（论文 §4.3.1：发布前人工增删敏感项）。

离线检测有 4–10% 转写错误 → 录音后主人可在标注界面**增删**敏感词。本模块定义
前端（M6）消费的 JSON 结构：一条音频一份 Annotation，含词流（含时间戳）与
敏感项列表；自动检测项 source="auto"，人工增补 source="manual"，删除走软删除
（deleted=True，保留审计痕迹）。加密时由 merge_spans() 把未删除项换算为
36ms WAV 帧闭区间（±余量帧，重叠合并）。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

SR = 16000
FRAME_LEN = 576          # 36ms @16kHz
MARGIN_FRAMES = 8        # 默认两侧余量（288ms）。FNR 扫参（experiments/eval_fnr.py，
                         # test-clean 60 句/160 敏感词）：8 帧 FNR=94.4%（论文 95.9%）
                         # 且 12/16 帧无收益——更长区间只增加整句损伤。

SCHEMA_VERSION = 1


@dataclass
class Word:
    text: str
    t0: float
    t1: float


@dataclass
class SensitiveItem:
    iid: str                    # 稳定 id（"w3-7" 或 "m1"）
    text: str
    category: str               # health/religion/PER/phone_digits/...
    t0: float
    t1: float
    source: str = "auto"        # auto | manual
    deleted: bool = False


@dataclass
class Annotation:
    audio: str                  # 相对路径或 URI
    words: list[Word]
    items: list[SensitiveItem] = field(default_factory=list)
    sr: int = SR
    frame_len: int = FRAME_LEN
    schema: int = SCHEMA_VERSION
    _manual_seq: int = 0

    # ── 构造 ──
    @classmethod
    def from_ctc(cls, audio: str,
                 ctc_words: list[tuple[str, float, float]]) -> "Annotation":
        return cls(audio=audio, words=[Word(w, t0, t1) for w, t0, t1 in ctc_words])

    def mark_words(self, indices: list[int], category: str,
                   source: str = "auto") -> SensitiveItem:
        """把连续词序号区间标记为一个敏感项。"""
        if not indices:
            raise ValueError("empty indices")
        i0, i1 = min(indices), max(indices)
        text = " ".join(self.words[i].text for i in range(i0, i1 + 1))
        item = SensitiveItem(f"w{i0}-{i1}", text, category,
                             self.words[i0].t0, self.words[i1].t1, source)
        self.items.append(item)
        return item

    def add_manual(self, text: str, t0: float, t1: float,
                   category: str = "manual") -> SensitiveItem:
        self._manual_seq += 1
        item = SensitiveItem(f"m{self._manual_seq}", text, category,
                             t0, t1, "manual")
        self.items.append(item)
        return item

    def remove(self, iid: str) -> None:
        for it in self.items:
            if it.iid == iid:
                it.deleted = True
                return
        raise KeyError(iid)

    def restore(self, iid: str) -> None:
        for it in self.items:
            if it.iid == iid:
                it.deleted = False
                return
        raise KeyError(iid)

    def set_range(self, iid: str, t0: float, t1: float) -> None:
        """手动调整起止时间（对齐误差修正）。"""
        for it in self.items:
            if it.iid == iid:
                it.t0, it.t1 = t0, t1
                return
        raise KeyError(iid)

    # ── 加密换算 ──
    def merge_spans(self, margin_frames: int = MARGIN_FRAMES,
                    ) -> list[tuple[int, int]]:
        """未删除敏感项 → 36ms 帧闭区间列表（±余量，重叠合并）。"""
        spans = []
        for it in self.items:
            if it.deleted:
                continue
            fs = int(it.t0 * self.sr / self.frame_len) - margin_frames
            fe = int(-(-it.t1 * self.sr // self.frame_len)) + margin_frames  # ceil
            spans.append((max(fs, 0), fe))
        spans.sort()
        merged = []
        for fs, fe in spans:
            if merged and fs <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], fe))
            else:
                merged.append((fs, fe))
        return merged

    # ── JSON ──
    def to_json(self, path: str | None = None) -> str:
        d = {
            "schema": self.schema, "audio": self.audio,
            "sr": self.sr, "frame_len": self.frame_len,
            "words": [asdict(w) for w in self.words],
            "items": [asdict(i) for i in self.items],
        }
        s = json.dumps(d, ensure_ascii=False, indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    @classmethod
    def from_json(cls, src: str) -> "Annotation":
        if src.strip().startswith("{"):
            d = json.loads(src)
        else:
            with open(src, encoding="utf-8") as f:
                d = json.load(f)
        if d["schema"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {d['schema']}")
        return cls(d["audio"], [Word(**w) for w in d["words"]],
                   [SensitiveItem(**i) for i in d["items"]],
                   sr=d["sr"], frame_len=d["frame_len"], schema=d["schema"])
