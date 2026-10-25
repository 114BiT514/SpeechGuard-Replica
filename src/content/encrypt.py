"""[M4 任务5] AES-CTR 帧级加密（论文 §4.3.2）。

内容单位是"词"：敏感词覆盖的帧区间（帧=36ms=576采样点@16k，int16 PCM）
逐帧加密。每个加密对象分配独立密钥 w_j（论文：c 个加密对象，密钥集 W）。

AES-CTR：流密码，密文与明文等长 → 帧结构不变，未加密部分播放无损（论文 §4.3.3
的前提）。无密钥者听到噪声（论文："encrypted segment may exhibit a harsher
sound"）；MP3 模式下无权限解码时替换为静音帧（见 mp3_crypto.py）。

nonce 约定：8 字节 = 加密对象序号 j（大端）；8 字节计数器从 0 起 → 同一密钥下
不同对象/不同区间的 keystream 互不重复。
"""
from __future__ import annotations

import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

FRAME_LEN = 576


def gen_key() -> bytes:
    """16 字节 AES-128 密钥。"""
    return get_random_bytes(16)


def _ctr(key: bytes, obj_index: int, data: bytes, decrypt: bool = False) -> bytes:
    nonce = obj_index.to_bytes(8, "big")
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce, initial_value=b"\0" * 8)
    return cipher.decrypt(data) if decrypt else cipher.encrypt(data)


def _apply(pcm: np.ndarray, spans, keys, decrypt: bool,
           frame_len: int = FRAME_LEN) -> np.ndarray:
    out = np.array(pcm, dtype=np.int16, copy=True)
    flat = out.view(np.uint8)
    for j, (fs, fe) in enumerate(spans):
        b0 = fs * frame_len * 2
        b1 = min((fe + 1) * frame_len * 2, flat.size)
        chunk = bytes(flat[b0:b1])
        res = _ctr(keys[j % len(keys)], j, chunk, decrypt=decrypt)
        flat[b0:b1] = np.frombuffer(res, dtype=np.uint8)
    return out


def encrypt_spans(pcm: np.ndarray, spans: list[tuple[int, int]], keys: list[bytes],
                  frame_len: int = FRAME_LEN) -> np.ndarray:
    """加密帧区间 [(fs, fe)]（闭区间帧号）。返回新数组，不修改输入。"""
    return _apply(pcm, spans, keys, decrypt=False, frame_len=frame_len)


def decrypt_spans(pcm: np.ndarray, spans: list[tuple[int, int]], keys: list[bytes],
                  frame_len: int = FRAME_LEN) -> np.ndarray:
    """解密（同区间同密钥，逐字节对称还原）。"""
    return _apply(pcm, spans, keys, decrypt=True, frame_len=frame_len)
