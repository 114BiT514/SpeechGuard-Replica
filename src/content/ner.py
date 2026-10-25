"""[M4 任务3] BERT NER（论文 §4.3.1：识别电话/地址等预定义"模式"对应的人名/
地名/机构等实体）。模型 dbmdz/bert-large-cased-finetuned-conll03-english
（已缓存 data/hf-cache），CoNLL03 标签：PER/LOC/ORG/MISC。
"""
from __future__ import annotations

import os
import re

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_ID = "dbmdz/bert-large-cased-finetuned-conll03-english"

_cache = {}


def load_ner(device: str = "cuda:0"):
    key = device
    if key in _cache:
        return _cache[key]
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    os.environ.setdefault("HF_HOME", os.path.join(PROJ, "data", "hf-cache"))
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID).to(device).eval()
    _cache[key] = (tok, model, torch)
    return _cache[key]


def extract_entities(text: str, device: str = "cuda:0",
                     labels=("PER", "LOC", "ORG", "MISC")):
    """返回 [(entity_text, label, start_char, end_char)]。

    子词聚合：每个 word 取其最后一个子词的预测标签（HF NER 惯例），
    连续同标签 word 合并为一个实体。字符位置由文本直接按词切分计算
    （该 tokenizer 不提供 offset_mapping）。
    """
    tok, model, torch = load_ner(device)
    enc = tok(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        preds = model(**enc).logits.argmax(-1)[0]
    names = [model.config.id2label.get(int(p), "O") for p in preds]
    word_ids = enc.word_ids(0)
    positions = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    last_sub = {}
    for i, w in enumerate(word_ids):
        if w is not None:
            last_sub[w] = i

    lab_of = {w: (names[i][2:] if names[i] != "O" else None)
              for w, i in last_sub.items()}

    out = []
    cur_lab, start_w = None, None
    ws = sorted(lab_of)
    for idx, w in enumerate(ws + [None]):
        lab = lab_of.get(w) if w is not None else None
        if lab != cur_lab:
            if cur_lab in labels and start_w is not None:
                prev_w = ws[idx - 1]
                s0, _ = positions[min(start_w, len(positions) - 1)]
                _, e0 = positions[min(prev_w, len(positions) - 1)]
                out.append((text[s0:e0], cur_lab, s0, e0))
            cur_lab, start_w = lab, w
    return out
