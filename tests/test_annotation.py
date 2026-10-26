"""M4 任务7 单测：手动标注 JSON 结构（增/删/改/换算）与检测流水线（词表+模式）。"""
import numpy as np

from src.content.annotation import Annotation
from src.content.detect import detect_sensitive
from src.content.sensitive_lists import SensitiveLists

WORDS = [
    ("MY", 0.00, 0.15),
    ("NAME", 0.15, 0.35),
    ("IS", 0.35, 0.45),
    ("DANIEL", 0.45, 0.75),      # NER 目标（本测试只测词表/模式路径）
    ("REED", 0.75, 0.95),
    ("LIVES", 0.95, 1.15),
    ("AT", 1.15, 1.25),
    ("FOUR", 1.25, 1.45),
    ("TWO", 1.45, 1.65),
    ("ORCHARD", 1.65, 1.95),
    ("LANE", 1.95, 2.15),
    ("CALL", 2.15, 2.30),
    ("FIVE", 2.30, 2.45),
    ("FIVE", 2.45, 2.60),
    ("FIVE", 2.60, 2.75),
    ("ZERO", 2.75, 2.90),
    ("ONE", 2.90, 3.05),
    ("TWO", 3.05, 3.20),
    ("SEVEN", 3.20, 3.40),
    ("NOW", 3.40, 3.55),
    ("SHE", 3.55, 3.70),
    ("IS", 3.70, 3.80),
    ("DIABETIC", 3.80, 4.20),
]


def test_detect_vocab_and_patterns():
    sl = SensitiveLists.build()
    hits = detect_sensitive(WORDS, sl, use_ner=False)
    idx_src = {h.idx: h.sources for h in hits}
    assert 22 in idx_src and "vocab:health" in idx_src[22]     # DIABETIC
    for i in range(12, 19):                                     # 7 位号码词全命中
        assert i in idx_src, i
        assert any(s.startswith("pattern:phone_words") for s in idx_src[i])
    assert 11 not in idx_src and 19 not in idx_src              # CALL/NOW 不牵连
    # 地址模式：LANE（idx 10）应命中
    assert 10 in idx_src
    assert any(s.startswith("pattern:address") for s in idx_src[10])


def test_detect_dedup_and_sources_tuple():
    hits = detect_sensitive(WORDS, SensitiveLists.build(), use_ner=False)
    assert len({h.idx for h in hits}) == len(hits)
    assert all(isinstance(h.sources, tuple) and h.sources for h in hits)


def _ann():
    return Annotation.from_ctc("data/demo.wav", WORDS)


def test_mark_and_merge_spans():
    ann = _ann()
    ann.mark_words([3, 4], "ner:PER")
    ann.mark_words([22], "vocab:health")
    spans = ann.merge_spans(margin_frames=0)
    assert len(spans) == 2
    # 帧号公式：fs=floor(t0*sr/576)，fe=ceil(t1*sr/576)
    def f(t):
        return int(np.floor(t * 16000 / 576)), int(-(-t * 16000 // 576))
    assert spans[0] == (f(0.45)[0], f(0.95)[1])
    assert spans[1] == (f(3.80)[0], f(4.20)[1])
    # 余量 + 合并
    sp12 = ann.merge_spans(margin_frames=12)
    assert sp12[0][0] == max(spans[0][0] - 12, 0) and sp12[0][1] == spans[0][1] + 12
    ann.mark_words([8, 9, 10], "pattern:address")
    sp = ann.merge_spans(margin_frames=6)
    assert len(sp) == 2   # (4-9) 与 (12-?) 合并成一个，DIABETIC 独立
    for (a0, b0), (a1, b1) in zip(sp, sp[1:]):
        assert a1 > b0 + 1  # 已重叠合并，不相邻


def test_manual_edit_operations():
    ann = _ann()
    it = ann.mark_words([3, 4], "ner:PER")
    m = ann.add_manual("MY PHONE", 2.0, 2.4)
    ann.remove(it.iid)
    assert ann.merge_spans(margin_frames=0) == [(int(2.0 * 16000 // 576),
                                                 int(-(-2.4 * 16000 // 576)))]
    ann.restore(it.iid)
    assert len(ann.merge_spans()) == 2
    ann.set_range(it.iid, 0.5, 0.8)
    sp = ann.merge_spans(margin_frames=0)
    a0, b0 = int(np.floor(0.5 * 16000 / 576)), int(-(-0.8 * 16000 // 576))
    a1, b1 = int(np.floor(2.0 * 16000 / 576)), int(-(-2.4 * 16000 // 576))
    assert sp == [(a0, b0), (a1, b1)]


def test_json_roundtrip(tmp_path):
    ann = _ann()
    ann.mark_words([3, 4], "ner:PER")
    ann.add_manual("备注：中文敏感项", 1.0, 1.5, category="custom")
    p = tmp_path / "anno.json"
    ann.to_json(str(p))
    b = Annotation.from_json(str(p))
    assert b.audio == ann.audio and len(b.words) == len(ann.words)
    assert [(i.iid, i.text, i.category, i.source, i.deleted, i.t0, i.t1)
            for i in b.items] == [(i.iid, i.text, i.category, i.source, i.deleted,
                                   i.t0, i.t1) for i in ann.items]
    assert b.merge_spans(margin_frames=5) == ann.merge_spans(margin_frames=5)
