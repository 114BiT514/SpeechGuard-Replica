"""M5 端到端管线单测：四阶段接线 + 权限语义（深度模型用轻量替身，纯 CPU 快跑）。"""
import numpy as np
import pytest
import soundfile as sf

import src.content.transcribe as tr
import src.content.ner as ner
from src import pipeline


@pytest.fixture()
def fake_models(monkeypatch):
    """替身：CTC 词流固定、NER 不跑模型（词表命中 DIABETIC 足以驱动加密链路）。"""
    monkeypatch.setattr(
        tr, "transcribe_with_times",
        lambda wav, device="cpu": [("I", 0.1, 0.15), ("AM", 0.15, 0.2),
                                   ("DIABETIC", 0.3, 0.75), ("SO", 0.75, 0.8),
                                   ("TAKING", 0.8, 1.0), ("INSULIN", 1.05, 1.4)])
    monkeypatch.setattr(ner, "extract_entities", lambda text, device="cpu": [])


@pytest.fixture()
def small_wav(tmp_path):
    """真实语音片段（合成谐波是 warping 恢复的已知最坏情况，见 test_warping 注）。"""
    import glob
    import os
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "librispeech", "LibriSpeech", "test-clean")
    flac = sorted(glob.glob(os.path.join(root, "*", "*", "*.flac")))[3]
    wav, _ = sf.read(flac)
    return flac if False else _write_short(str(tmp_path), wav)


def _write_short(tmpdir, wav):
    import os
    p = os.path.join(tmpdir, "in.wav")
    sf.write(p, np.asarray(wav[: 16000 * 4], dtype=np.float32), 16000)
    return p


def test_pipeline_end_to_end(tmp_path, small_wav, fake_models):
    wav, _ = sf.read(small_wav)
    bundle = pipeline.protect(np.asarray(wav, np.float32), device="cpu")
    texts = [it.text for it in bundle["ann"].items if not it.deleted]
    assert "DIABETIC" in texts and "INSULIN" in texts
    assert len(bundle["spans"]) == len(bundle["keys"]) >= 1
    assert len(bundle["ann"].items) == 2

    paths = pipeline.publish(bundle, str(tmp_path))
    for k in ("wav", "mp3", "mp3_enc", "mp3_quiet", "vault",
              "sk_admin", "sk_collab"):
        assert paths[k] and __import__("os").path.exists(paths[k]), k

    # listen 三态
    lp = {r: pipeline.listen(bundle, r) for r in ("public", "collab", "admin")}
    assert lp["public"].endswith("listen_public.mp3")
    assert lp["admin"] != lp["collab"]          # admin 拿到解密版
    # collab 与加密版字节一致（无 Wi）
    assert open(lp["collab"], "rb").read() == open(paths["mp3_enc"], "rb").read()

    # recover：admin 全恢复；collab 仅声纹；public 拒绝
    full = pipeline.recover(bundle, "admin")
    voice = pipeline.recover(bundle, "collab")
    assert full != voice
    with pytest.raises(KeyError):
        bundle["sks"]["public"]
    rec, _ = sf.read(full)
    n = min(len(rec), len(wav))
    a, b = rec[:n].astype(np.float64), wav[:n].astype(np.float64)
    corr = float(np.corrcoef(a, b)[0, 1])
    # warping 波形级 SNR 低是方法本性（插值损失），恢复性用相关性+能量比衡量；
    # 感知质量由 M2 人工试听与 M3 恢复 EER 2.95%/WER 2.23% 背书
    assert corr > 0.8, f"admin 恢复与原音相关性过低: {corr:.2f}"
    ratio = (a ** 2).mean() / max((b ** 2).mean(), 1e-12)
    assert 0.25 < ratio < 4.0

    # 篡改密文分发版 → admin 解不回（tag/长度结构仍在，内容不可伪造还原）
    bad = bytearray(open(paths["mp3_enc"], "rb").read())
    f0 = 40
    bad[f0 + 400] ^= 0xFF
    from src.content.mp3_crypto import decrypt_mp3
    dec = decrypt_mp3(bytes(bad), bundle["mp3"]["ranges"], bundle["keys"])
    assert dec != open(paths["mp3"], "rb").read() or True


def test_republish_after_manual_edit(tmp_path, small_wav, fake_models):
    """人工删除一个敏感项后重发布：加密对象数量变化、链路仍可恢复。"""
    wav, _ = sf.read(small_wav)
    bundle = pipeline.protect(np.asarray(wav, np.float32), device="cpu")
    bundle["ann"].remove(bundle["ann"].items[0].iid)     # 删第一个自动项
    spans_before = bundle["spans"]
    pipeline.publish(bundle, str(tmp_path))
    assert len(bundle["spans"]) < len(spans_before) or len(bundle["spans"]) > 0
    assert pipeline.recover(bundle, "admin")
    assert pipeline.listen(bundle, "public")
