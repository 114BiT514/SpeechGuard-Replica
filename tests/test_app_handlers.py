"""[P0-4] 前端 handler 报错文案与增删逻辑单测（不经 gradio 校验层，直达处理函数）。"""
import numpy as np
import pytest
import soundfile as sf

from app import main as app
from src import pipeline

CTC = [("WE", 0.5, 0.65), ("DISPUTE", 1.0, 1.4),
       ("POLITICS", 1.64, 2.04), ("RELIGION", 2.2, 3.1)]


@pytest.fixture()
def bundle(monkeypatch):
    import src.content.transcribe as tr
    import src.content.ner as ner
    monkeypatch.setattr(tr, "transcribe_with_times", lambda wav, device="cpu": CTC)
    monkeypatch.setattr(ner, "extract_entities", lambda text, device="cpu": [])
    wav = (0.2 * np.sin(2 * np.pi * 300 * np.arange(16000 * 3.2) / 16000)
           ).astype(np.float32)
    return pipeline.protect(wav, device="cpu")


def test_add_words_error_messages(bundle):
    with pytest.raises(Exception) as e:
        app.add_words(None, "3: DISPUTE", None)
    assert "①执行保护" in str(e.value)
    with pytest.raises(Exception) as e:
        app.add_words(bundle, None, None)
    assert "起始词" in str(e.value)


def test_remove_item_error_messages(bundle):
    with pytest.raises(Exception) as e:
        app.remove_item(bundle, None)
    assert "①执行保护" in str(e.value) or "删除敏感项" in str(e.value)
    with pytest.raises(Exception) as e:
        app.remove_item(None, "w2-2")
    assert "①执行保护" in str(e.value)


def test_add_then_remove_flow(bundle, tmp_path, monkeypatch):
    bundle["tmpdir"] = str(tmp_path)          # 真实 app 里由 protect_detect 建
    assert len([i for i in bundle["ann"].items if not i.deleted]) == 1  # 相邻词合并为一项
    # 增补 DISPUTE（idx 1）
    b, md, _, _ = app.add_words(bundle, "1: DISPUTE [1.00-1.40]", None)
    assert "DISPUTE" in md and md.count("manual") >= 1
    n_items = len([i for i in b["ann"].items if not i.deleted])
    assert n_items == 2
    # 删除它 → 软删除留痕
    b2, md2, _, _ = app.remove_item(b, "w1-1: DISPUTE")
    assert "已删除" in md2
    assert len([i for i in b2["ann"].items if not i.deleted]) == 1
    # 加密区间随之收缩（DISPUTE 1.0-1.4 若独立成段则段数变化）
    assert b2["spans"] == b2["ann"].merge_spans()
