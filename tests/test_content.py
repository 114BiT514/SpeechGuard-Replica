"""M4 内容保护模块单元测试：敏感词表 + AES-CTR 帧加密。"""
import numpy as np

from src.content.encrypt import FRAME_LEN, decrypt_spans, encrypt_spans, gen_key
from src.content.sensitive_lists import SensitiveLists


def test_gdpr_word_match():
    sl = SensitiveLists.build()
    assert sl.match_word("DIABETIC") == "health"
    assert sl.match_word("church") == "religion"
    assert sl.match_word("ELECTIONS,") == "politics"   # 标点剥离
    assert sl.match_word("TABLE") is None


def test_custom_words():
    sl = SensitiveLists.build(custom_words=["QUILTER", "rochester"])
    assert sl.match_word("QUILTER") == "custom"
    assert sl.match_word("ROCHESTER") == "custom"


def test_patterns():
    sl = SensitiveLists.build()
    assert sl.match_pattern("CALL FIVE FIVE FIVE ZERO ONE TWO SEVEN NOW")  # 电话词
    hits = sl.match_pattern("BORN ON JUNE THE THIRD IN")
    assert hits and hits[0][2] == "date"
    assert not sl.match_pattern("THE WEATHER IS FINE TODAY")


def test_encrypt_roundtrip_exact():
    """加密→解密逐字节还原；未加密区间不动。"""
    rng = np.random.default_rng(0)
    pcm = (rng.standard_normal(FRAME_LEN * 10) * 8000).astype(np.int16)
    keys = [gen_key(), gen_key()]
    spans = [(2, 3), (6, 8)]
    enc = encrypt_spans(pcm, spans, keys)
    # 未加密帧完全不变
    np.testing.assert_array_equal(enc[: 2 * FRAME_LEN], pcm[: 2 * FRAME_LEN])
    np.testing.assert_array_equal(enc[4 * FRAME_LEN: 6 * FRAME_LEN],
                                  pcm[4 * FRAME_LEN: 6 * FRAME_LEN])
    # 加密帧变了
    assert not np.array_equal(enc[2 * FRAME_LEN: 4 * FRAME_LEN],
                              pcm[2 * FRAME_LEN: 4 * FRAME_LEN])
    # 解密还原
    dec = decrypt_spans(enc, spans, keys)
    np.testing.assert_array_equal(dec, pcm)


def test_wrong_key_garbles():
    rng = np.random.default_rng(1)
    pcm = (rng.standard_normal(FRAME_LEN * 4) * 8000).astype(np.int16)
    k1, k2 = gen_key(), gen_key()
    enc = encrypt_spans(pcm, [(1, 2)], [k1])
    dec = decrypt_spans(enc, [(1, 2)], [k2])
    assert not np.array_equal(dec[FRAME_LEN: 3 * FRAME_LEN], pcm[FRAME_LEN: 3 * FRAME_LEN])


def test_length_preserved():
    """CTR 流密码：密文与明文等长（帧结构不变，MP3 适配的前提）。"""
    pcm = (np.random.default_rng(2).standard_normal(FRAME_LEN * 7) * 5000).astype(np.int16)
    enc = encrypt_spans(pcm, [(0, 6)], [gen_key()])
    assert enc.shape == pcm.shape and enc.dtype == np.int16
